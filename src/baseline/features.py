"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from . import config, constants

def _load_nomic_model():
    """Load Nomic model once with optimized settings."""
    model = SentenceTransformer(
        config.NOMIC_MODEL_NAME,
        trust_remote_code=True,
        device=config.NOMIC_DEVICE
    )
    # Disable truncation warnings
    model.max_seq_length = config.NOMIC_MAX_LENGTH
    return model


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.

    Uses the training data to compute mean ratings and interaction counts
    to prevent data leakage from the test set.

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

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")


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
    # Create a mapping book_id -> description
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    # Get descriptions for books in df (in the same order)
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    # Transform to TF-IDF features
    tfidf_matrix = vectorizer.transform(df_descriptions)

    # Convert sparse matrix to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    # Concatenate TF-IDF features with main DataFrame
    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(tfidf_feature_names)} TF-IDF features.")
    return df_with_tfidf


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
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

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill TF-IDF features with 0 (for books without descriptions)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features with 0 (for books without descriptions)
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df

def add_book_and_author_embeddings(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    descriptions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds Nomic embeddings for books AND authors in one pass.
    Caches book embeddings → computes author averages from train only.
    """
    print("Adding book and author Nomic embeddings...")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    book_emb_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME
    author_emb_path = config.MODEL_DIR / "author_nomic_embeddings.pkl"

    # Prepare descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
    all_descriptions = all_descriptions.drop_duplicates(subset=constants.COL_BOOK_ID)

    # === BOOK EMBEDDINGS ===
    if book_emb_path.exists():
        print(f"Loading cached book embeddings from {book_emb_path}")
        book_embeddings_dict = joblib.load(book_emb_path)
    else:
        print("Computing book Nomic embeddings...")
        model = _load_nomic_model()

        book_ids = all_descriptions[constants.COL_BOOK_ID].tolist()
        texts = ["search_document: " + desc for desc in all_descriptions[constants.COL_DESCRIPTION].tolist()]

        # Batch encoding with progress bar
        embeddings = []
        batch_size = config.NOMIC_BATCH_SIZE
        for i in tqdm(range(0, len(texts), batch_size), desc="Book embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_emb = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            embeddings.extend(batch_emb)

        book_embeddings_dict = dict(zip(book_ids, embeddings, strict=False))
        joblib.dump(book_embeddings_dict, book_emb_path)
        print(f"Book embeddings saved to {book_emb_path}")

    # === AUTHOR EMBEDDINGS (from train books only) ===
    if author_emb_path.exists():
        print(f"Loading cached author embeddings from {author_emb_path}")
        author_embeddings_dict = joblib.load(author_emb_path)
    else:
        print("Computing author embeddings (train only)...")
        train_books = train_df[constants.COL_BOOK_ID].unique()
        train_desc = all_descriptions[all_descriptions[constants.COL_BOOK_ID].isin(train_books)]

        # Merge with book -> author mapping
        book_author_map = df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]].drop_duplicates()
        train_desc_with_author = train_desc.merge(book_author_map, on=constants.COL_BOOK_ID, how="left")

        # Group by author and average book embeddings
        author_emb_list = []
        author_ids = []

        for author_id, group in tqdm(train_desc_with_author.groupby(constants.COL_AUTHOR_ID), desc="Author avg"):
            book_ids_in_group = group[constants.COL_BOOK_ID].tolist()
            valid_embs = [book_embeddings_dict[bid] for bid in book_ids_in_group if bid in book_embeddings_dict]
            if valid_embs:
                avg_emb = np.mean(valid_embs, axis=0)
            else:
                avg_emb = np.zeros(config.NOMIC_EMBEDDING_DIM)
            author_emb_list.append(avg_emb)
            author_ids.append(author_id)

        author_embeddings_dict = dict(zip(author_ids, author_emb_list, strict=False))
        joblib.dump(author_embeddings_dict, author_emb_path)
        print(f"Author embeddings saved to {author_emb_path}")

    # === MAP TO DF ===
    # Book embeddings
    book_emb_matrix = np.array([
        book_embeddings_dict.get(bid, np.zeros(config.NOMIC_EMBEDDING_DIM))
        for bid in df[constants.COL_BOOK_ID]
    ])
    book_cols = [f"nomic_book_{i}" for i in range(config.NOMIC_EMBEDDING_DIM)]
    book_df = pd.DataFrame(book_emb_matrix, columns=book_cols, index=df.index)

    # Author embeddings
    author_emb_matrix = np.array([
        author_embeddings_dict.get(aid, np.zeros(config.NOMIC_EMBEDDING_DIM))
        for aid in df[constants.COL_AUTHOR_ID]
    ])
    author_cols = [f"nomic_author_{i}" for i in range(config.NOMIC_EMBEDDING_DIM)]
    author_df = pd.DataFrame(author_emb_matrix, columns=author_cols, index=df.index)

    # Concat
    df = pd.concat([df.reset_index(drop=True), book_df, author_df], axis=1)
    print(f"Added {len(book_cols) + len(author_cols)} Nomic features (book + author).")
    return df


def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = add_book_and_author_embeddings(df, train_df, descriptions_df)  # ← НОВАЯ ФУНКЦИЯ
    df = handle_missing_values(df, train_df)

    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df

