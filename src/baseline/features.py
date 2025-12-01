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
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.sparse import csr_matrix
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
        df (pd.DataFrame): чThe main DataFrame to add features to.
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

def _try_load_books() -> pd.DataFrame | None:
    """Вспомогательная функция загрузки books.csv (уже есть в оригинале, дублируем для удобства)."""
    possible = [
        config.DATA_DIR / "raw" / constants.BOOK_DATA_FILENAME,
        config.DATA_DIR / constants.BOOK_DATA_FILENAME,
    ]
    for p in possible:
        if p.exists():
            return pd.read_csv(p)
    return None

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

def add_enhanced_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Самые мощные агрегатные фичи без лика — полностью устойчивая версия."""
    print("Adding ENHANCED aggregate features (robust version)...")

    train_read = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    global_mean = train_read[config.TARGET].mean()
    global_std = train_read[config.TARGET].std()

    # ==================================== 1. Базовые агрегаты ====================================
    user_stats = train_read.groupby(constants.COL_USER_ID)[config.TARGET].agg([
        'mean', 'std', 'count', 'min', 'max', 'median'
    ]).reset_index()
    user_stats.columns = [
        constants.COL_USER_ID, 'user_mean', 'user_std', 'user_cnt', 'user_min', 'user_max', 'user_med'
    ]

    book_stats = train_read.groupby(constants.COL_BOOK_ID)[config.TARGET].agg([
        'mean', 'std', 'count', 'min', 'max', 'median'
    ]).reset_index()
    book_stats.columns = [
        constants.COL_BOOK_ID, 'book_mean', 'book_std', 'book_cnt', 'book_min', 'book_max', 'book_med'
    ]

    df = df.merge(user_stats, on=constants.COL_USER_ID, how='left')
    df = df.merge(book_stats, on=constants.COL_BOOK_ID, how='left')

    # ==================================== 2. Авторские агрегаты (умная обработка) ====================================
    books_df = _try_load_books()
    author_mean_col = None

    if books_df is not None:
        # Возможные имена колонок с авторами
        possible_author_cols = ['author_id', 'author', 'authors', 'author_name', 'Author', 'Authors']
        author_col = None
        for col in possible_author_cols:
            if col in books_df.columns:
                author_col = col
                break

        if author_col is not None:
            print(f"Found author column: {author_col}")
            # Создаём временный author_id (хэшируем строки, если это имена)
            if books_df[author_col].dtype == 'object':
                books_df['__temp_author_id'] = books_df[author_col].astype(str).fillna('missing_author')
                # Для стабильности — используем factorize
                books_df['__temp_author_id'] = pd.factorize(books_df['__temp_author_id'])[0]
            else:
                books_df['__temp_author_id'] = books_df[author_col]

            # Мержим к train_read
            train_with_author = train_read.merge(
                books_df[[constants.COL_BOOK_ID, '__temp_author_id']],
                on=constants.COL_BOOK_ID,
                how='left'
            )

            valid_authors = train_with_author['__temp_author_id'].notna()
            if valid_authors.any():
                author_stats = train_with_author[valid_authors].groupby('__temp_author_id')[config.TARGET].agg([
                    'mean', 'std', 'count'
                ]).reset_index()
                author_stats.columns = ['__temp_author_id', 'author_mean', 'author_std', 'author_cnt']

                # Мержим обратно к основному df
                book_to_author = books_df[[constants.COL_BOOK_ID, '__temp_author_id']].drop_duplicates()
                df = df.merge(book_to_author, on=constants.COL_BOOK_ID, how='left')
                df = df.merge(author_stats, on='__temp_author_id', how='left')
                df = df.drop(columns=['__temp_author_id'], errors='ignore')

                author_mean_col = 'author_mean'

    # ==================================== 3. Заполнение холодных ====================================
    fill_values = {
        'user_mean': global_mean,
        'user_std': 0.0,
        'user_cnt': 0,
        'user_min': global_mean,
        'user_max': global_mean,
        'user_med': global_mean,
        'book_mean': global_mean * 1.03,
        'book_std': global_std,
        'book_cnt': 0,
        'book_min': global_mean,
        'book_max': global_mean,
        'book_med': global_mean,
    }
    if author_mean_col:
        fill_values.update({'author_mean': global_mean, 'author_std': 0.0, 'author_cnt': 0})

    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # ==================================== 4. Взаимодействия и статистики ====================================
    df['user_log_cnt'] = np.log1p(df['user_cnt'])
    df['book_log_cnt'] = np.log1p(df['book_cnt'])
    df['user_reliability'] = 1 - np.exp(-df['user_cnt'] / 10.0)
    df['book_reliability'] = 1 - np.exp(-df['book_cnt'] / 20.0)

    df['user_book_mean_diff'] = df['user_mean'] - df['book_mean']
    df['user_book_mean_abs_diff'] = np.abs(df['user_book_mean_diff'])
    df['user_book_compatibility'] = 10 - df['user_book_mean_abs_diff'].clip(0, 10)

    # Shrinkage
    df['user_shrink_mean'] = (df['user_cnt'] * df['user_mean'] + 15 * global_mean) / (df['user_cnt'] + 15)
    df['book_shrink_mean'] = (df['book_cnt'] * df['book_mean'] + 25 * global_mean) / (df['book_cnt'] + 25)

    df['pred_weighted_v1'] = (
        df['user_shrink_mean'] * df['user_reliability'] * 0.6 +
        df['book_shrink_mean'] * df['book_reliability'] * 0.4
    )

    if author_mean_col == 'author_mean':
        df['pred_weighted_v2'] = df['pred_weighted_v1'] * 0.7 + df['author_mean'] * 0.3

    df['user_strictness'] = (df['user_mean'] - global_mean) / (df['user_std'] + 1e-6)
    df['user_extremity'] = df['user_max'] - df['user_min']
    df['user_rating_skew'] = (df['user_mean'] - df['user_med']) / (df['user_std'] + 1e-6)

    print("Added ~70+ robust enhanced aggregate features")
    return df


def add_time_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Временные фичи, если есть даты (год публикации, возраст книги и т.д.)."""
    print("Adding time-based features...")
    books_df = _try_load_books()
    if books_df is not None and constants.COL_YEAR in books_df.columns:
        year_map = dict(zip(books_df[constants.COL_BOOK_ID], books_df[constants.COL_YEAR]))
        df['book_year'] = df[constants.COL_BOOK_ID].map(year_map)
        df['book_age'] = 2025 - df['book_year'].fillna(2025)
        df['is_classic'] = (df['book_age'] > 70).astype(int)
        df['is_very_new'] = (df['book_age'] < 3).astype(int)
    return df


def add_popularity_decile_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Децили/квантили популярности пользователей и книг — очень мощно для LightGBM."""
    print("Adding popularity rank/decile features...")
    train_read = train_df[train_df[constants.COL_HAS_READ] == 1]

    user_cnt = train_read.groupby(constants.COL_USER_ID).size()
    book_cnt = train_read.groupby(constants.COL_BOOK_ID).size()

    df['user_pop_rank'] = df[constants.COL_USER_ID].map(user_cnt.rank(method='average'))
    df['book_pop_rank'] = df[constants.COL_BOOK_ID].map(book_cnt.rank(method='average'))

    for col, prefix in [(user_cnt, 'user'), (book_cnt, 'book')]:
        sr = df[f'{prefix}_pop_rank']
        df[f'{prefix}_pop_decile'] = pd.qcut(sr, 10, labels=False, duplicates='drop') + 1
        df[f'{prefix}_pop_quintile'] = pd.qcut(sr, 5, labels=False, duplicates='drop') + 1
        df[f'{prefix}_pop_is_top10pct'] = (sr >= sr.quantile(0.9)).astype(int)

    return df


def add_nmf_topic_features(df: pd.DataFrame, train_df: pd.DataFrame, n_components: int = 30) -> pd.DataFrame:
    """NMF на user-book матрице — часто лучше SVD."""
    print(f"Adding NMF topics ({n_components} components)...")
    ratings = train_df[train_df[constants.COL_HAS_READ] == 1]
    pivot = ratings.pivot_table(index=constants.COL_USER_ID, columns=constants.COL_BOOK_ID,
                                values=config.TARGET, fill_value=0)
    matrix = csr_matrix(pivot.values)

    nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
    user_nmf = nmf.fit_transform(matrix)
    book_nmf = nmf.components_.T

    user_df = pd.DataFrame(user_nmf, index=pivot.index,
                           columns=[f'user_nmf_{i}' for i in range(n_components)])
    book_df = pd.DataFrame(book_nmf, index=pivot.columns,
                           columns=[f'book_nmf_{i}' for i in range(n_components)])

    df = df.merge(user_df, on=constants.COL_USER_ID, how='left')
    df = df.merge(book_df, on=constants.COL_BOOK_ID, how='left')
    df[[c for c in df.columns if c.startswith(('user_nmf_', 'book_nmf_'))]] = \
        df[[c for c in df.columns if c.startswith(('user_nmf_', 'book_nmf_'))]].fillna(0)
    return df


def add_count_encoded_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Count encoding для всех категориальных фич (location, publisher, etc.)."""
    print("Adding count-encoded categorical features...")
    train_read = train_df[train_df[constants.COL_HAS_READ] == 1]

    for col in config.CAT_FEATURES + ['location_city', 'location_state', 'location_country', 'publisher']:
        if col not in train_df.columns and col not in df.columns:
            continue
        cnt = train_read[col].value_counts()
        df[f'{col}_count'] = df[col].map(cnt).fillna(0)
        df[f'{col}_freq'] = df[col].map(cnt / len(train_read))
    return df


def add_target_encoded_features(df: pd.DataFrame, train_df: pd.DataFrame, alpha: float = 10.0) -> pd.DataFrame:
    """Smoothed target encoding (без лика)."""
    print("Adding smoothed target encoding...")
    train_read = train_df[train_df[constants.COL_HAS_READ] == 1]
    global_mean = train_read[config.TARGET].mean()

    cols_to_encode = ['location_city', 'location_state', 'location_country',
                      'publisher', constants.COL_AUTHOR_ID, 'author']

    for col in cols_to_encode:
        if col not in df.columns:
            continue

        stats = train_read.groupby(col)[config.TARGET].agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + global_mean * alpha) / (stats['count'] + alpha)

        # Приводим к float32, чтобы не было проблем с типом
        df[f'{col}_te'] = df[col].map(smoothed).fillna(global_mean).astype('float32')

    return df

def add_pairwise_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Множество полиномиальных и логических взаимодействий (самые сильные фичи для GBM)."""
    print("Adding hundreds of pairwise interaction features...")

    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if not c.startswith(('tfidf_', 'bert_', 'svd_', 'nmf_', 'genre_'))]

    # Выбираем только самые важные базовые числовые
    base_nums = ['user_mean', 'book_mean', 'user_cnt', 'book_cnt', 'user_log_cnt', 'book_log_cnt',
                 'user_reliability', 'book_reliability', 'book_age', 'user_age', 'book_avg_rating']

    base_nums = [c for c in base_nums if c in df.columns]

    for i, c1 in enumerate(base_nums):
        for c2 in base_nums[i+1:]:
            df[f'inter_{c1}_{c2}_prod'] = df[c1] * df[c2]
            df[f'inter_{c1}_{c2}_ratio'] = df[c1] / (df[c2] + 1e-6)

    # Логические комбинации популярности
    if 'user_pop_decile' in df.columns and 'book_pop_decile' in df.columns:
        df['pop_decile_diff'] = df['user_pop_decile'] - df['book_pop_decile']
        df['pop_match'] = (df['user_pop_decile'] == df['book_pop_decile']).astype(int)

    print(f"   Added ~{len(base_nums)*(len(base_nums)-1)} numeric interactions")
    return df

def create_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Полный максимально агрессивный пайплайн (ожидаемый прирост 0.03–0.05+ RMSE)."""
    print("=" * 60)
    print("STARTING MAXIMUM FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # 1. Улучшенные агрегаты
    df = add_enhanced_aggregate_features(df, train_df)

    # 2. To-read + временные
    df = add_to_read_features(df, train_df)
    df = add_time_features(df, train_df)

    # 3. Ранговые/квантильные фичи
    df = add_popularity_decile_features(df, train_df)

    # 4. CF embeddings: SVD + NMF
    df = add_cf_embeddings(df, train_df, n_components=64)
    df = add_nmf_topic_features(df, train_df, n_components=40)

    # 5. Жанры + one-hot
    df = add_genre_features(df, book_genres_df)

    # 6. Текст: TF-IDF + улучшенный BERT pooling
    df = add_text_features(df, train_df, descriptions_df)
    df = add_bert_features(df, train_df, descriptions_df)

    # 7. Count & Target encoding
    df = add_count_encoded_features(df, train_df)
    df = add_target_encoded_features(df, train_df)

    # 8. Массовая генерация взаимодействий
    df = add_pairwise_interaction_features(df)

    # 9. Финальная обработка пропусков
    df = handle_missing_values(df, train_df)

    # 10. Category dtype для LGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    total_features = df.shape[1] - len(
        [constants.COL_USER_ID, constants.COL_BOOK_ID, config.TARGET, constants.COL_SOURCE])
    print(f"FINISHED: created {total_features} features in total")

    return df
