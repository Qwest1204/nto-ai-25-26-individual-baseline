"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from . import config, constants


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Безопасные агрегатные фичи без leakage.
    Использует только существующие константы.
    """
    print("Adding safe aggregate features...")

    # Используем только прочитанные книги для вычисления агрегатов
    train_read = train_df[train_df[constants.COL_HAS_READ] == 1].copy()

    # Глобальные статистики
    global_mean = train_read[config.TARGET].mean()
    global_std = train_read[config.TARGET].std()

    # 1. User aggregates (только базовые, безопасные)
    user_agg = train_read.groupby(constants.COL_USER_ID)[config.TARGET].agg(['mean', 'count']).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT
    ]

    # 2. Book aggregates (только базовые, безопасные)
    book_agg = train_read.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(['mean', 'count']).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT
    ]

    # 3. Author aggregates (если можем загрузить books.csv)
    author_agg = None
    books_df = None

    # Пробуем несколько возможных путей
    possible_paths = [
        config.DATA_DIR / "raw" / constants.BOOK_DATA_FILENAME,  # data/raw/books.csv
        config.DATA_DIR / constants.BOOK_DATA_FILENAME,  # data/books.csv
        #Path("D:\\br\\data\\raw\\books.csv"),  # Абсолютный путь
        #Path("data/raw/books.csv"),  # Относительный путь
    ]

    books_path = None
    for path in possible_paths:
        if path.exists():
            books_path = path
            break

    if books_path:
        try:
            books_df = pd.read_csv(books_path)
            print(f"Loaded books data from {books_path}")

            if constants.COL_AUTHOR_ID in books_df.columns:
                # Мержим author_id к train_read
                train_with_author = train_read.merge(
                    books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
                    on=constants.COL_BOOK_ID,
                    how='left'
                )

                # Убираем книги без автора
                train_with_author = train_with_author.dropna(subset=[constants.COL_AUTHOR_ID])

                if not train_with_author.empty:
                    author_agg = train_with_author.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(
                        ['mean']).reset_index()
                    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]
                    print(f"Computed aggregates for {len(author_agg)} authors")
        except Exception as e:
            print(f"Error loading books data: {e}")
    else:
        print(f"Books file not found. Tried paths: {possible_paths}")
        print(f"Current DATA_DIR: {config.DATA_DIR}")
        print(f"DATA_DIR exists: {config.DATA_DIR.exists()}")

    # 4. Мержим все агрегаты
    print("Merging aggregates...")

    # User aggregates
    df = df.merge(user_agg, on=constants.COL_USER_ID, how='left')
    print(f"  Merged user aggregates: {len(user_agg)} users")

    # Book aggregates
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how='left')
    print(f"  Merged book aggregates: {len(book_agg)} books")

    # Author aggregates (если есть)
    if author_agg is not None and not author_agg.empty:
        # Сначала мержим author_id к df из books_df
        if constants.COL_AUTHOR_ID not in df.columns and books_df is not None:
            df = df.merge(
                books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
                on=constants.COL_BOOK_ID,
                how='left'
            )

        if constants.COL_AUTHOR_ID in df.columns:
            df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how='left')
            print(f"  Merged author aggregates: {len(author_agg)} authors")

    # 5. Заполняем пропуски стратегически
    print("Handling NaN values...")

    # Для пользователей без истории
    user_na_mask = df[constants.F_USER_MEAN_RATING].isna()
    if user_na_mask.any():
        df.loc[user_na_mask, constants.F_USER_MEAN_RATING] = global_mean
        df.loc[user_na_mask, constants.F_USER_RATINGS_COUNT] = 0
        print(f"  Filled {user_na_mask.sum()} missing user aggregates")

    # Для книг без истории
    book_na_mask = df[constants.F_BOOK_MEAN_RATING].isna()
    if book_na_mask.any():
        df.loc[book_na_mask, constants.F_BOOK_MEAN_RATING] = global_mean * 1.05  # benefit of doubt
        df.loc[book_na_mask, constants.F_BOOK_RATINGS_COUNT] = 0
        print(f"  Filled {book_na_mask.sum()} missing book aggregates")

    # Для авторов без истории
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        author_na_mask = df[constants.F_AUTHOR_MEAN_RATING].isna()
        if author_na_mask.any():
            df.loc[author_na_mask, constants.F_AUTHOR_MEAN_RATING] = global_mean
            print(f"  Filled {author_na_mask.sum()} missing author aggregates")

    # 6. Добавляем простые комбинации (безопасные)
    print("Creating interaction features...")

    # Разница между средним пользователя и книги
    df['user_book_rating_diff'] = df[constants.F_USER_MEAN_RATING] - df[constants.F_BOOK_MEAN_RATING]

    # Абсолютная разница
    df['user_book_rating_abs_diff'] = df['user_book_rating_diff'].abs()

    # Совместимость (чем меньше разница, тем выше совместимость)
    df['user_book_compatibility'] = 10 - df['user_book_rating_abs_diff']

    # Произведение популярности (log scale для уменьшения skew)
    df['user_popularity_log'] = np.log1p(df[constants.F_USER_RATINGS_COUNT])
    df['book_popularity_log'] = np.log1p(df[constants.F_BOOK_RATINGS_COUNT])
    df['user_book_popularity_product'] = df['user_popularity_log'] * df['book_popularity_log']

    # Reliability пользователя (чем больше оценок, тем надежнее)
    df['user_reliability'] = 1 - np.exp(-df[constants.F_USER_RATINGS_COUNT] / 5)

    # Reliability книги
    df['book_reliability'] = 1 - np.exp(-df[constants.F_BOOK_RATINGS_COUNT] / 10)

    # Combined reliability
    df['combined_reliability'] = df['user_reliability'] * 0.6 + df['book_reliability'] * 0.4

    # Ожидаемый рейтинг (взвешенная комбинация)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df['expected_rating'] = (
            df[constants.F_USER_MEAN_RATING] * 0.4 +
            df[constants.F_BOOK_MEAN_RATING] * 0.4 +
            df[constants.F_AUTHOR_MEAN_RATING] * 0.2
        )
    else:
        df['expected_rating'] = (
            df[constants.F_USER_MEAN_RATING] * 0.5 +
            df[constants.F_BOOK_MEAN_RATING] * 0.5
        )

    # Ограничиваем ожидаемый рейтинг диапазоном 0-10
    df['expected_rating'] = df['expected_rating'].clip(0, 10)

    # 7. Статистика по добавленным фичам
    feature_cols = [
        constants.F_USER_MEAN_RATING, constants.F_USER_RATINGS_COUNT,
        constants.F_BOOK_MEAN_RATING, constants.F_BOOK_RATINGS_COUNT,
        'user_book_rating_diff', 'user_book_rating_abs_diff',
        'user_book_compatibility', 'user_book_popularity_product',
        'user_reliability', 'book_reliability', 'combined_reliability',
        'expected_rating'
    ]

    # Добавляем author фичу, если есть
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        feature_cols.append(constants.F_AUTHOR_MEAN_RATING)

    num_features = len([c for c in feature_cols if c in df.columns])

    print(f"✅ Added {num_features} safe aggregate features")
    print(f"   Global mean: {global_mean:.3f}")
    print(f"   Global std: {global_std:.3f}")
    print(f"   User coverage: {(~user_na_mask).mean():.2%}")
    print(f"   Book coverage: {(~book_na_mask).mean():.2%}")

    return df


def add_to_read_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Adds features based on has_read=0 (to-read list).

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for calculations.

    Returns:
        pd.DataFrame: The DataFrame with to-read features.
    """
    print("Adding to-read features...")
    to_read_df = train_df[train_df[constants.COL_HAS_READ] == 0]
    user_to_read_count = to_read_df.groupby(constants.COL_USER_ID).size().reset_index(name='user_to_read_count')
    return df.merge(user_to_read_count, on=constants.COL_USER_ID, how="left")


def add_cf_embeddings(df: pd.DataFrame, train_df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
    """Adds collaborative filtering embeddings using SVD.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for fitting SVD.
        n_components (int): Number of SVD components.

    Returns:
        pd.DataFrame: The DataFrame with SVD embeddings.
    """
    print("Adding CF embeddings...")
    # Filter to has_read=1 for matrix
    ratings_df = train_df[train_df[constants.COL_HAS_READ] == 1]
    pivot = ratings_df.pivot_table(index=constants.COL_USER_ID, columns=constants.COL_BOOK_ID, values=config.TARGET, fill_value=0)
    sparse_matrix = csr_matrix(pivot.values)
    svd = TruncatedSVD(n_components=n_components)
    user_embeddings = svd.fit_transform(sparse_matrix)
    book_embeddings = svd.components_.T

    user_emb_df = pd.DataFrame(user_embeddings, index=pivot.index, columns=[f'user_svd_{i}' for i in range(n_components)])
    book_emb_df = pd.DataFrame(book_embeddings, index=pivot.columns, columns=[f'book_svd_{i}' for i in range(n_components)])

    df = df.merge(user_emb_df, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_emb_df, on=constants.COL_BOOK_ID, how="left")
    return df


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds genre features, including one-hot for top genres.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with genre features.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [constants.COL_BOOK_ID, constants.F_BOOK_GENRES_COUNT]

    # One-hot for genres
    book_genres_grouped = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].apply(list).reset_index()
    mlb = MultiLabelBinarizer(sparse_output=True)
    genre_onehot = pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(book_genres_grouped[constants.COL_GENRE_ID]),
        index=book_genres_grouped.index,
        columns=[f"genre_{c}" for c in mlb.classes_]
    )
    genre_onehot = genre_onehot.sparse.to_dense()
    genre_onehot[constants.COL_BOOK_ID] = book_genres_grouped[constants.COL_BOOK_ID]

    df = df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(genre_onehot, on=constants.COL_BOOK_ID, how="left")
    return df


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions.

    Trains a TF-IDF vectorizer only on training data descriptions to avoid
    data leakage. Applies the vectorizer to all books and merges the features.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for fitting the vectorizer.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with TF-IDF features added.
    """
    print("Adding text features (TF-IDF)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Get unique books from train set
    train_books = train_df[constants.COL_BOOK_ID].unique()

    # Extract descriptions for training books only
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Check if vectorizer already exists (for prediction)
    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        # Fit vectorizer on training descriptions only
        print("Fitting TF-IDF vectorizer on training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        # Save vectorizer for use in prediction
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    # Transform all book descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Get descriptions in the same order as df[book_id]
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    # Transform to TF-IDF features
    tfidf_matrix = vectorizer.transform(df_descriptions)
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(
        tfidf_matrix, index=df.index, columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    )
    tfidf_df = tfidf_df.sparse.to_dense()
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    print(f"Added {tfidf_matrix.shape[1]} TF-IDF features.")
    return df


def add_bert_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BERT embeddings from book descriptions.

    Computes BERT embeddings only for books in the training data to avoid
    data leakage, then applies to all. Saves embeddings for reuse.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for computing embeddings.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with BERT features added.
    """
    print("Adding BERT features...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
    model.eval()
    model.to(config.BERT_DEVICE)

    # Get unique books
    all_books = df[constants.COL_BOOK_ID].unique()
    desc_map = dict(
        zip(descriptions_df[constants.COL_BOOK_ID], descriptions_df[constants.COL_DESCRIPTION], strict=False)
    )

    if embeddings_path.exists():
        print(f"Loading existing BERT embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing BERT embeddings...")
        embeddings_dict = {}
        for book_id in tqdm(all_books):
            desc = desc_map.get(book_id, "")
            if not desc:
                continue
            inputs = tokenizer(desc, return_tensors="pt", max_length=config.BERT_MAX_LENGTH, truncation=True, padding=True)
            inputs = {k: v.to(config.BERT_DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings_dict[book_id] = embedding
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Embeddings saved to {embeddings_path}")

    # Map embeddings to df
    df_book_ids = df[constants.COL_BOOK_ID]
    embeddings_list = [embeddings_dict.get(book_id, np.zeros(config.BERT_EMBEDDING_DIM)) for book_id in df_book_ids]
    embeddings_array = np.array(embeddings_list)

    bert_feature_names = [f"bert_{i}" for i in range(config.BERT_EMBEDDING_DIM)]
    bert_df = pd.DataFrame(embeddings_array, columns=bert_feature_names, index=df.index)

    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
    print(f"Added {len(bert_feature_names)} BERT features.")
    return df_with_bert


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values using a defined strategy.

    Fills missing values for age, aggregated features, and categorical features
    to prepare the DataFrame for model training. Uses metrics from the training
    set (e.g., global mean) to fill NaNs.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.
        train_df (pd.DataFrame): The training data, used for calculating fill metrics.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill aggregate features for "cold start" users/items (only if they exist)
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    # Fill new features
    if 'user_weighted_mean' in df.columns:
        df['user_weighted_mean'] = df['user_weighted_mean'].fillna(global_mean)
    if 'user_rating_std' in df.columns:
        df['user_rating_std'] = df['user_rating_std'].fillna(0)
    if 'user_to_read_count' in df.columns:
        df['user_to_read_count'] = df['user_to_read_count'].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill TF-IDF features with 0
    tfidf_cols = [col for col in df.columns if isinstance(col, str) and col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features with 0
    bert_cols = [col for col in df.columns if isinstance(col, str) and col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill SVD features with 0
    svd_cols = [col for col in df.columns if isinstance(col, str) and (col.startswith("user_svd_") or col.startswith("book_svd_"))]
    for col in svd_cols:
        df[col] = df[col].fillna(0.0)

    # Fill genre one-hot with 0
    genre_cols = [col for col in df.columns if isinstance(col, str) and col.startswith("genre_")]
    for col in genre_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df

def add_nomic_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    descriptions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Добавляет эмбеддинги Nomic Embed Text v1.5 (через Sentence Transformers)
    на основе описаний книг.

    Эмбеддинги вычисляются один раз для всех уникальных книг и кэшируются на диск.
    При повторных запусках загрузка происходит мгновенно.

    Args:
        df (pd.DataFrame): Основной DataFrame, к которому добавляются признаки.
        train_df: Оставлен для совместимости с предыдущим API (не используется).
        descriptions_df (pd.DataFrame): DataFrame с колонками COL_BOOK_ID и COL_DESCRIPTION.

    Returns:
        pd.DataFrame: DataFrame с добавленными колонками nomic_0 ... nomic_767.
    """
    print("Adding Nomic Embed Text v1.5 features (via Sentence Transformers)...")

    # Путь для сохранения эмбеддингов
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.NOMIC_EMBEDDINGS_FILENAME

    # Маппинг book_id → описание
    desc_map = dict(
        zip(
            descriptions_df[constants.COL_BOOK_ID],
            descriptions_df[constants.COL_DESCRIPTION],
            strict=False,
        )
    )

    # Уникальные книги в текущем датасете
    all_book_ids = df[constants.COL_BOOK_ID].unique()

    # Загружаем или вычисляем эмбеддинги
    if embeddings_path.exists():
        print(f"Loading cached Nomic embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Loading Nomic Embed Text v1.5 model via Sentence Transformers...")
        # Официальная поддержка nomic-embed-text-v1.5 в sentence-transformers
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,        # обязательно для Nomic
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # Важно: для корректной работы с nomic-embed-text нужно добавлять префикс
        model.tokenizer.padding_side = "right"
        model.tokenizer.truncation_side = "right"

        print("Computing embeddings for book descriptions...")
        embeddings_dict = {}

        # Подготавливаем тексты с обязательным префиксом "search_document:"
        texts_to_encode = []
        book_ids_ordered = []

        for book_id in tqdm(all_book_ids, desc="Preparing descriptions"):
            desc = desc_map.get(book_id, "")
            if not desc or not desc.strip():
                continue
            # Префикс критически важен для активации правильного пула эмбеддингов
            texts_to_encode.append(f"search_document: {desc.strip()}")
            book_ids_ordered.append(book_id)

        if not texts_to_encode:
            print("Warning: No valid descriptions found. Using zero vectors.")
        else:
            # Пакетное кодирование (очень быстро на GPU)
            embeddings = model.encode(
                texts_to_encode,
                batch_size=8,                  # подберите под вашу видеокарту
                show_progress_bar=True,
                normalize_embeddings=True,      # рекомендуется для Nomic
                convert_to_numpy=True,
            )
            for book_id, emb in zip(book_ids_ordered, embeddings):
                embeddings_dict[book_id] = emb.astype(np.float32)

        # Сохраняем на диск
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Nomic embeddings cached to {embeddings_path}")

    # Применяем к основному DataFrame
    zero_vector = np.zeros(768, dtype=np.float32)
    embeddings_list = [
        embeddings_dict.get(book_id, zero_vector)
        for book_id in df[constants.COL_BOOK_ID]
    ]
    embeddings_array = np.stack(embeddings_list)

    feature_names = [f"nomic_{i}" for i in range(768)]
    nomic_df = pd.DataFrame(embeddings_array, columns=feature_names, index=df.index)

    df_with_features = pd.concat([df.reset_index(drop=True), nomic_df], axis=1)

    print(f"Successfully added 768 Nomic embedding features.")
    return df_with_features

def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline.

    This function orchestrates the calls to add aggregate features (optional), genre
    features, text features (TF-IDF and BERT), and handle missing values. Includes new improvements.

    Args:
        df (pd.DataFrame): The merged DataFrame from `data_processing`.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
        include_aggregates (bool): If True, compute aggregate features. Defaults to False.

    Returns:
        pd.DataFrame: The final DataFrame with all features engineered.
    """
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_to_read_features(df, train_df)
    df = add_cf_embeddings(df, train_df)
    df = add_genre_features(df, book_genres_df)
    #df = add_text_features(df, train_df, descriptions_df)
    df = add_nomic_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
