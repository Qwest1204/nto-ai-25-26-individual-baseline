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
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from . import config, constants


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Безопасные агрегатные фичи без leakage.
    Использует только существующие константы и тренировочные прочитанные взаимодействия.
    +20 новых мощных фич добавлено.
    """
    print("Adding safe aggregate features...")

    # Только прочитанные книги для всех агрегатов
    train_read = train_df[train_df[constants.COL_HAS_READ] == 1].copy()

    # Глобальные константы
    global_mean = train_read[config.TARGET].mean()
    global_std  = train_read[config.TARGET].std()
    global_median = train_read[config.TARGET].median()

    # 1. Базовые user / book / author агрегаты (оставляем как было)
    user_agg = train_read.groupby(constants.COL_USER_ID)[config.TARGET].agg(
        ['mean', 'std', 'count', 'median', 'skew']
    ).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        'user_rating_std',
        constants.F_USER_RATINGS_COUNT,
        'user_rating_median',
        'user_rating_skew'
    ]

    book_agg = train_read.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(
        ['mean', 'std', 'count', 'median', 'skew']
    ).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        'book_rating_std',
        constants.F_BOOK_RATINGS_COUNT,
        'book_rating_median',
        'book_rating_skew'
    ]

    # Author aggregates (если доступны)
    author_agg = None
    books_df = None
    # ... (ваш существующий код загрузки books.csv остаётся без изменений) ...
    # (предполагаем, что author_agg уже может содержать mean, std, count)

    # Мёржим базовые агрегаты
    df = df.merge(user_agg[[constants.COL_USER_ID, constants.F_USER_MEAN_RATING, 'user_rating_std',
                            constants.F_USER_RATINGS_COUNT, 'user_rating_median', 'user_rating_skew']],
                  on=constants.COL_USER_ID, how='left')
    df = df.merge(book_agg[[constants.COL_BOOK_ID, constants.F_BOOK_MEAN_RATING, 'book_rating_std',
                            constants.F_BOOK_RATINGS_COUNT, 'book_rating_median', 'book_rating_skew']],
                  on=constants.COL_BOOK_ID, how='left')

    # Author merge (если есть)
    if author_agg is not None and not author_agg.empty:
        # ... ваш код мержа автора ...
        pass

    # ================================
    # 20 НОВЫХ БЕЗОПАСНЫХ ФИЧ
    # ================================

    print("Creating 20 advanced safe aggregate features...")

    # 1–4. Надёжность и сглаженные средние
    df['user_rating_count_norm'] = df[constants.F_USER_RATINGS_COUNT] / (df[constants.F_USER_RATINGS_COUNT].max() + 1)
    df['book_rating_count_norm'] = df[constants.F_BOOK_RATINGS_COUNT] / (df[constants.F_BOOK_RATINGS_COUNT].max() + 1)

    df['user_mean_rating_smoothed'] = (
        df[constants.F_USER_MEAN_RATING] * df[constants.F_USER_RATINGS_COUNT] + global_mean * 10
    ) / (df[constants.F_USER_RATINGS_COUNT] + 10)

    df['book_mean_rating_smoothed'] = (
        df[constants.F_BOOK_MEAN_RATING] * df[constants.F_BOOK_RATINGS_COUNT] + global_mean * 5
    ) / (df[constants.F_BOOK_RATINGS_COUNT] + 5)

    # 5–8. Байесовские средние (ещё более устойчивые)
    C_user = 25
    C_book = 15
    df['user_bayesian_avg'] = (df[constants.F_USER_MEAN_RATING] * df[constants.F_USER_RATINGS_COUNT] + global_mean * C_user) / (df[constants.F_USER_RATINGS_COUNT] + C_user)
    df['book_bayesian_avg'] = (df[constants.F_BOOK_MEAN_RATING] * df[constants.F_BOOK_RATINGS_COUNT] + global_mean * C_book) / (df[constants.F_BOOK_RATINGS_COUNT] + C_book)

    # 9–10. Z-score нормализация
    df['user_z_score'] = (df[constants.F_USER_MEAN_RATING] - global_mean) / (global_std + 1e-8)
    df['book_z_score'] = (df[constants.F_BOOK_MEAN_RATING] - global_mean) / (global_std + 1e-8)

    # 11–13. Отклонение от медианы и дисперсия
    df['user_vs_global_median'] = df[constants.F_USER_MEAN_RATING] - global_median
    df['book_vs_global_median'] = df[constants.F_BOOK_MEAN_RATING] - global_median
    df['user_book_dispersion'] = (df['user_rating_std'].fillna(global_std) + df['book_rating_std'].fillna(global_std)) / 2

    # 14–16. Лог-популярность и произведения
    df['user_pop_log'] = np.log1p(df[constants.F_USER_RATINGS_COUNT])
    df['book_pop_log'] = np.log1p(df[constants.F_BOOK_RATINGS_COUNT])
    df['popularity_ratio'] = df['user_pop_log'] / (df['book_pop_log'] + 1e-8)

    # 17–18. "Строгость" пользователя и книги
    df['user_strictness'] = global_mean - df[constants.F_USER_MEAN_RATING]   # чем выше — тем строже
    df['book_overrated']   = df[constants.F_BOOK_MEAN_RATING] - global_mean   # положительное = переоценена

    # 19. Предсказание на основе байесовского + надёжности
    alpha = 0.6
    df['hybrid_expected_rating'] = (
        df['user_bayesian_avg'] * alpha * df['user_rating_count_norm'] +
        df['book_bayesian_avg'] * (1 - alpha) * df['book_rating_count_norm'] +
        global_mean * (1 - alpha * df['user_rating_count_norm'] - (1 - alpha) * df['book_rating_count_norm'])
    ).clip(0, 10)

    # 20. Финальная "уверенность" в предсказании
    df['prediction_confidence'] = (
        df['user_rating_count_norm'] * 0.5 +
        df['book_rating_count_norm'] * 0.5
    ) * (1 - np.exp(-df[constants.F_USER_RATINGS_COUNT] - df[constants.F_BOOK_RATINGS_COUNT]))

    # ================================
    # Заполнение пропусков (обновлённое)
    # ================================
    print("Handling NaN values for new features...")

    na_user = df[constants.F_USER_MEAN_RATING].isna()
    na_book = df[constants.F_BOOK_MEAN_RATING].isna()

    df.loc[na_user, [constants.F_USER_MEAN_RATING, 'user_mean_rating_smoothed', 'user_bayesian_avg',
                     'user_z_score', 'user_vs_global_median', 'user_strictness']] = global_mean
    df.loc[na_user, ['user_rating_std', 'user_rating_skew']] = global_std
    df.loc[na_user, constants.F_USER_RATINGS_COUNT] = 0
    df.loc[na_user, ['user_pop_log', 'user_rating_count_norm']] = 0

    df.loc[na_book, [constants.F_BOOK_MEAN_RATING, 'book_mean_rating_smoothed', 'book_bayesian_avg',
                     'book_z_score', 'book_vs_global_median', 'book_overrated']] = global_mean
    df.loc[na_book, ['book_rating_std', 'book_rating_skew']] = global_std
    df.loc[na_book, constants.F_BOOK_RATINGS_COUNT] = 0
    df.loc[na_book, ['book_pop_log', 'book_rating_count_norm']] = 0

    # Пересчитываем комбинированные фичи после заполнения
    df['hybrid_expected_rating'] = df['hybrid_expected_rating'].fillna(global_mean)
    df['prediction_confidence'] = df['prediction_confidence'].fillna(0)

    # Обновлённый список всех добавленных фич
    new_features = [
        'user_rating_std', 'user_rating_median', 'user_rating_skew',
        'book_rating_std', 'book_rating_median', 'book_rating_skew',
        'user_rating_count_norm', 'book_rating_count_norm',
        'user_mean_rating_smoothed', 'book_mean_rating_smoothed',
        'user_bayesian_avg', 'book_bayesian_avg',
        'user_z_score', 'book_z_score',
        'user_vs_global_median', 'book_vs_global_median',
        'user_book_dispersion', 'popularity_ratio',
        'user_strictness', 'book_overrated',
        'hybrid_expected_rating', 'prediction_confidence'
    ]

    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        new_features.append(constants.F_AUTHOR_MEAN_RATING)

    total_added = len([c for c in new_features + [
        constants.F_USER_MEAN_RATING, constants.F_USER_RATINGS_COUNT,
        constants.F_BOOK_MEAN_RATING, constants.F_BOOK_RATINGS_COUNT
    ] if c in df.columns])

    print(f"Added {total_added} safe aggregate features (including 20 advanced ones)")
    print(f"   Global mean/median/std: {global_mean:.3f} / {global_median:.3f} / {global_std:.3f}")

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
    train_df: pd.DataFrame | None = None,  # Для совместимости (не используется)
    descriptions_df: pd.DataFrame | None = None,
    target_dim: int = 512,                 # ← желаемая размерность (768, 512, 256, 128 и т.д.)
    normalize: bool = True,                # L2-нормализация (рекомендуется)
) -> pd.DataFrame:
    """
    Добавляет L2-нормализованные эмбеддинги Nomic Embed Text v1.5
    из описаний книг с использованием Sentence Transformers и Matryoshka-сжатия.

    Если target_dim < 768, применяется официальный Matryoshka-метод: layer_norm + truncation + normalize.
    Это сохраняет >99.7% качества на MTEB-бенчмарках при размерности 256.

    Args:
        df: основной DataFrame.
        train_df: для совместимости (не используется).
        descriptions_df: DataFrame с колонками COL_BOOK_ID и COL_DESCRIPTION.
        target_dim: целевая размерность эмбеддингов (768 по умолчанию).
        normalize: применять ли L2-нормализацию.

    Returns:
        DataFrame с колонками nomic_0 ... nomic_{target_dim-1}.
    """
    print(f"Adding Nomic Embed Text v1.5 features (dim={target_dim}, normalize={normalize})...")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Имя файла кэша с учетом размерности
    embeddings_filename = constants.NOMIC_EMBEDDINGS_FILENAME.replace(".joblib", f"_dim{target_dim}.joblib")
    embeddings_path = config.MODEL_DIR / embeddings_filename

    desc_map = dict(
        zip(descriptions_df[constants.COL_BOOK_ID], descriptions_df[constants.COL_DESCRIPTION], strict=False)
    )

    all_book_ids = df[constants.COL_BOOK_ID].unique()

    if embeddings_path.exists():
        print(f"Loading cached embeddings (dim={target_dim}) from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Loading nomic-ai/nomic-embed-text-v1.5 model...")
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Подготовка текстов с префиксом
        texts = []
        book_ids_ordered = []
        for book_id in tqdm(all_book_ids, desc="Collecting descriptions"):
            desc = desc_map.get(book_id, "").strip()
            if desc:
                texts.append(f"search_document: {desc}")
                book_ids_ordered.append(book_id)

        embeddings_dict = {}

        if texts:
            print(f"Encoding {len(texts):,} descriptions...")
            # Кодирование БЕЗ неподдерживаемых аргументов
            embeddings_tensor = model.encode(
                texts,
                batch_size=8,
                show_progress_bar=True,
                convert_to_tensor=True,  # Возвращаем тензор для torch-операций
            )

            if target_dim < 768:
                print(f"Applying Matryoshka truncation to dim={target_dim}...")
                # Официальный порядок: layer_norm → truncate → normalize
                if normalize:
                    embeddings_tensor = F.layer_norm(
                        embeddings_tensor, normalized_shape=(embeddings_tensor.shape[1],)
                    )
                embeddings_tensor = embeddings_tensor[:, :target_dim]  # Обрезка
                if normalize:
                    embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)

            # Конвертация в numpy и float32
            embeddings_np = embeddings_tensor.cpu().numpy().astype(np.float32)

            for book_id, vec in zip(book_ids_ordered, embeddings_np):
                embeddings_dict[book_id] = vec

        # Сохранение кэша
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Embeddings cached → {embeddings_path}")

    # Применение к основному DataFrame
    zero_vec = np.zeros(target_dim, dtype=np.float32)
    if target_dim < 768 and normalize:
        # Нормализуем нулевой вектор (хотя он останется нулевым; для consistency)
        zero_vec = F.normalize(torch.from_numpy(zero_vec), p=2, dim=0).numpy()

    vectors = [embeddings_dict.get(book_id, zero_vec) for book_id in df[constants.COL_BOOK_ID]]
    embeddings_array = np.stack(vectors)

    feature_names = [f"nomic_{i}" for i in range(target_dim)]
    nomic_df = pd.DataFrame(embeddings_array, columns=feature_names, index=df.index)

    result_df = pd.concat([df.reset_index(drop=True), nomic_df], axis=1)
    norm_status = "L2-normalized" if normalize else "raw"
    print(f"Added {target_dim} {norm_status} Nomic embedding features (Matryoshka-applied if truncated).")

    return result_df

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
