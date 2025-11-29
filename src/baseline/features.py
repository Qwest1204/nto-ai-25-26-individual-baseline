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

import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from . import config, constants


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.

    Uses the training data to compute mean ratings and interaction counts
    to prevent data leakage from the test set. Includes time-weighted means and std.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion of the data for calculations.

    Returns:
        pd.DataFrame: The DataFrame with new aggregate features.
    """
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Weighted mean rating (exponential decay)
    train_df['time_decay'] = np.exp(-(pd.to_datetime(train_df[constants.COL_TIMESTAMP]) - train_df[constants.COL_TIMESTAMP].min()).dt.days / 30)
    user_weighted_mean = train_df.groupby(constants.COL_USER_ID).apply(
        lambda x: np.average(x[config.TARGET], weights=x['time_decay']), include_groups=False
    ).reset_index(name='user_weighted_mean')

    # Std deviation of ratings
    user_std = train_df.groupby(constants.COL_USER_ID)[config.TARGET].std().reset_index(name='user_rating_std')

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")
    df = df.merge(user_weighted_mean, on=constants.COL_USER_ID, how="left")
    df = df.merge(user_std, on=constants.COL_USER_ID, how="left")

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
    """Adds BERT-like embeddings from book descriptions using the configured model (e.g., Nomic).

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for calculations (unused here).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with BERT/Nomic features.
    """
    print("Computing embeddings...")
    embeddings_path = config.INTERIM_DATA_DIR / "bert_embeddings.pkl"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    if embeddings_path.exists():
        embeddings_dict = joblib.load(embeddings_path)
        print(f"Loaded embeddings from {embeddings_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME, trust_remote_code=True).to(config.BERT_DEVICE)

        unique_book_ids = df[constants.COL_BOOK_ID].unique()
        embeddings_dict = {}

        for book_id in tqdm(unique_book_ids):
            description = descriptions_df.loc[descriptions_df[constants.COL_BOOK_ID] == book_id, 'description'].values
            if len(description) == 0 or pd.isna(description[0]):
                embeddings_dict[book_id] = np.zeros(config.BERT_EMBEDDING_DIM)
                continue

            text = f"search_document: {description[0].strip()}"  # Nomic prefix for document embeddings
            inputs = tokenizer(text, return_tensors="pt", max_length=config.BERT_MAX_LENGTH, truncation=True, padding=True)
            inputs = {k: v.to(config.BERT_DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

            # Apply layer normalization if the model supports it (Nomic-specific)
            if hasattr(model, 'layernorm'):
                embedding = model.layernorm(embedding)

            # L2 normalization as per Nomic recommendations
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).squeeze().cpu().numpy()
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
    print(f"Added {len(bert_feature_names)} embedding features.")
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


def add_categorical_embeddings(df: pd.DataFrame, train_df: pd.DataFrame, embedding_dim: int = 32) -> pd.DataFrame:
    """Генерирует эмбеддинги для категориальных признаков с помощью autoencoder.

    Обучает на train_df, применяет к df. Поддерживает high-cardinality categoricals.

    Args:
        df: Основной DataFrame.
        train_df: Train для обучения.
        embedding_dim: Размерность эмбеддинга на признак.

    Returns:
        DataFrame с добавленными эмбеддингами.
    """
    print("Adding categorical embeddings...")
    cat_cols = [col for col in config.CAT_FEATURES if col in train_df.columns and train_df[col].nunique() > 2]

    for col in tqdm(cat_cols):
        # Manual encoding to handle unseen
        train_vals = train_df[col].astype(str).fillna('missing')
        unique_vals = train_vals.unique()
        val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
        vocab_size = len(unique_vals) + 1  # +1 for unseen category

        # Encode train (0 to len-1)
        train_encoded = np.array([val_to_idx[val] for val in train_vals])

        # Encode df, unseen to len(unique_vals)
        df_vals = df[col].astype(str).fillna('missing')
        df_encoded = np.array([val_to_idx.get(val, len(unique_vals)) for val in df_vals])

        # Autoencoder with Embedding
        class Autoencoder(nn.Module):
            def __init__(self, vocab_size, embedding_dim):
                super().__init__()
                self.encoder = nn.Embedding(vocab_size, embedding_dim)
                self.decoder = nn.Linear(embedding_dim, vocab_size)

            def forward(self, x):
                emb = self.encoder(x)
                logits = self.decoder(emb)
                return logits

        model = Autoencoder(vocab_size, embedding_dim).to(config.BERT_DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Dataset for train
        train_tensor = torch.tensor(train_encoded, dtype=torch.long)
        dataset = TensorDataset(train_tensor, train_tensor)  # input and target are the same indices
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in tqdm(range(10)):
            for inputs, targets in tqdm(loader):
                inputs, targets = inputs.to(config.BERT_DEVICE), targets.to(config.BERT_DEVICE)
                logits = model(inputs)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Get embeddings for df
        with torch.no_grad():
            df_tensor = torch.tensor(df_encoded, dtype=torch.long).to(config.BERT_DEVICE)
            embeddings = model.encoder(df_tensor).cpu().numpy()

        emb_cols = [f"{col}_emb_{i}" for i in range(embedding_dim)]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
        df = pd.concat([df, emb_df], axis=1)

    return df


# Новая функция для числовых эмбеддингов (PCA для компрессии)
def add_numeric_embeddings(df: pd.DataFrame, train_df: pd.DataFrame, n_components: int = 16) -> pd.DataFrame:
    """Компрессирует числовые признаки в эмбеддинги с помощью PCA.

    Args:
        df: Основной DataFrame.
        train_df: Train для fit.
        n_components: Количество компонент.

    Returns:
        DataFrame с добавленными эмбеддингами.
    """
    print("Adding numeric embeddings...")
    num_cols = [col for col in train_df.columns if
                pd.api.types.is_numeric_dtype(train_df[col]) and col != config.TARGET]

    if not num_cols:
        return df

    scaler = StandardScaler().fit(train_df[num_cols].fillna(0))
    train_scaled = scaler.transform(train_df[num_cols].fillna(0))
    df_scaled = scaler.transform(df[num_cols].fillna(0))

    pca = PCA(n_components=n_components).fit(train_scaled)
    embeddings = pca.transform(df_scaled)

    emb_cols = [f"num_emb_{i}" for i in range(n_components)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
    return pd.concat([df, emb_df], axis=1)


# Обновите create_features для интеграции
def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_to_read_features(df, train_df)
    df = add_cf_embeddings(df, train_df)
    df = add_genre_features(df, book_genres_df)
    #df = add_text_features(df, train_df, descriptions_df)
    df = add_bert_features(df, train_df, descriptions_df)

    # Новые эмбеддинги
    df = add_categorical_embeddings(df, train_df)
    df = add_numeric_embeddings(df, train_df)

    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
