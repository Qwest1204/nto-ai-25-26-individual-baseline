# features.py (полный код с улучшениями)

"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

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
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count", "std"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
        'user_rating_std'
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count", "std"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
        'book_rating_std'
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean", "std"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING, 'author_rating_std']

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")

    # New: Interactions with std
    df['user_book_std_diff'] = df['user_rating_std'] - df['book_rating_std']
    return df


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

    # New: Genre mean ratings (target encoding for genres)
    train_genres = book_genres_df.merge(train_df[[constants.COL_BOOK_ID, config.TARGET]], on=constants.COL_BOOK_ID, how='inner')
    genre_mean = train_genres.groupby(constants.COL_GENRE_ID)[config.TARGET].mean().reset_index()
    genre_mean.columns = [constants.COL_GENRE_ID, 'genre_mean_rating']

    # Agg genre means per book (mean of genre means)
    book_genre_means = book_genres_df.merge(genre_mean, on=constants.COL_GENRE_ID).groupby(constants.COL_BOOK_ID)['genre_mean_rating'].mean().reset_index()
    book_genre_means.columns = [constants.COL_BOOK_ID, 'book_genre_mean_rating']

    df = df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(book_genre_means, on=constants.COL_BOOK_ID, how="left")
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

    # Apply vectorizer to all descriptions
    all_descriptions = descriptions_df.copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
    tfidf_matrix = vectorizer.transform(all_descriptions[constants.COL_DESCRIPTION])

    # Create DataFrame with TF-IDF features
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
    )
    tfidf_df[constants.COL_BOOK_ID] = all_descriptions[constants.COL_BOOK_ID].values

    # Merge TF-IDF features into main DataFrame
    df = df.merge(tfidf_df, on=constants.COL_BOOK_ID, how="left")
    print(f"Added {tfidf_matrix.shape[1]} TF-IDF features.")
    return df


def add_book_and_author_embeddings(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds Nomic embeddings for books and authors.

    Fits embeddings on training data only, reduces dimensionality with PCA.
    Averages author embeddings across their books.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for fitting embeddings.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with embedding features added.
    """
    print("Adding Nomic embeddings (book + author)...")
    TARGET_EMBED_DIM = 384  # Increased from 300

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    book_emb_path = config.MODEL_DIR / "nomic_book_embeddings_384.pkl"
    author_emb_path = config.MODEL_DIR / "nomic_author_embeddings_384.pkl"

    # Load or compute book embeddings
    if book_emb_path.exists():
        print(f"Loading existing book embeddings from {book_emb_path}")
        book_embeddings_dict = joblib.load(book_emb_path)
    else:
        model = _load_nomic_model()

        # Get unique books from train (for fit PCA)
        train_books = train_df[constants.COL_BOOK_ID].unique()
        train_desc = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
        train_desc[constants.COL_DESCRIPTION] = train_desc[constants.COL_DESCRIPTION].fillna("No description available.")

        # Compute embeddings in batches
        print("Computing book embeddings on train...")
        train_embs = []
        for i in tqdm(range(0, len(train_desc), config.NOMIC_BATCH_SIZE)):
            batch = train_desc[constants.COL_DESCRIPTION].iloc[i:i + config.NOMIC_BATCH_SIZE].tolist()
            train_embs.extend(model.encode(batch, show_progress_bar=False))

        train_embs = np.array(train_embs)

        # Fit PCA on train embeddings
        pca = PCA(n_components=TARGET_EMBED_DIM)
        pca.fit(train_embs)

        # Now compute for all books
        all_desc = descriptions_df.copy()
        all_desc[constants.COL_DESCRIPTION] = all_desc[constants.COL_DESCRIPTION].fillna("No description available.")

        print("Computing book embeddings on all...")
        all_embs = []
        for i in tqdm(range(0, len(all_desc), config.NOMIC_BATCH_SIZE)):
            batch = all_desc[constants.COL_DESCRIPTION].iloc[i:i + config.NOMIC_BATCH_SIZE].tolist()
            all_embs.extend(model.encode(batch, show_progress_bar=False))

        all_embs = np.array(all_embs)

        # Apply PCA
        reduced_embs = pca.transform(all_embs)

        book_embeddings_dict = dict(zip(all_desc[constants.COL_BOOK_ID], reduced_embs))
        joblib.dump(book_embeddings_dict, book_emb_path)
        print(f"Book embeddings (384-dim) saved to {book_emb_path}")

    # Author embeddings: average book embeddings per author
    if author_emb_path.exists():
        print(f"Loading existing author embeddings from {author_emb_path}")
        author_embeddings_dict = joblib.load(author_emb_path)
    else:
        # Merge author_id to descriptions
        author_desc = descriptions_df.merge(df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]].drop_duplicates(), on=constants.COL_BOOK_ID)

        unique_authors = author_desc[constants.COL_AUTHOR_ID].unique()
        author_emb_list = []
        author_ids = []

        print("Computing author embeddings (avg book emb)...")
        for author_id in tqdm(unique_authors):
            author_books = author_desc[author_desc[constants.COL_AUTHOR_ID] == author_id][constants.COL_BOOK_ID]
            author_embs = np.array([book_embeddings_dict.get(bid, np.zeros(TARGET_EMBED_DIM)) for bid in author_books])
            if len(author_embs) > 0:
                avg_emb = np.mean(author_embs, axis=0)
            else:
                avg_emb = np.zeros(TARGET_EMBED_DIM)
            author_emb_list.append(avg_emb)
            author_ids.append(author_id)

        author_embeddings_dict = dict(zip(author_ids, author_emb_list))
        joblib.dump(author_embeddings_dict, author_emb_path)
        print(f"Author embeddings (384-dim) saved to {author_emb_path}")

    # Map to df
    # Book embeddings
    book_emb_matrix = np.array([
        book_embeddings_dict.get(bid, np.zeros(TARGET_EMBED_DIM))
        for bid in df[constants.COL_BOOK_ID]
    ])
    book_cols = [f"nomic_book_{i}" for i in range(TARGET_EMBED_DIM)]
    book_df = pd.DataFrame(book_emb_matrix, columns=book_cols, index=df.index)

    # Author embeddings
    author_emb_matrix = np.array([
        author_embeddings_dict.get(aid, np.zeros(TARGET_EMBED_DIM))
        for aid in df[constants.COL_AUTHOR_ID]
    ])
    author_cols = [f"nomic_author_{i}" for i in range(TARGET_EMBED_DIM)]
    author_df = pd.DataFrame(author_emb_matrix, columns=author_cols, index=df.index)

    # Concat
    df = pd.concat([df.reset_index(drop=True), book_df, author_df], axis=1)
    print(f"Added {len(book_cols) + len(author_cols)} Nomic features (book + author, 384-dim).")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    if constants.COL_TIMESTAMP in df.columns:
        df['day_of_week'] = df[constants.COL_TIMESTAMP].dt.dayofweek.astype('category')
        df['month'] = df[constants.COL_TIMESTAMP].dt.month.astype('category')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Recency: days since user's first rating (approx)
        df['user_first_ts'] = df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].transform('min')
        df['days_since_first'] = (df[constants.COL_TIMESTAMP] - df['user_first_ts']).dt.days.fillna(0)
        df.drop('user_first_ts', axis=1, inplace=True)  # Clean up
    return df


def add_similarity_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between user pref embeddings and book/author embeddings."""
    # User prefs: mean book embedding per user from train
    book_emb_cols = [f"nomic_book_{i}" for i in range(384)]
    user_book_emb = train_df.merge(df[[constants.COL_BOOK_ID] + book_emb_cols], on=constants.COL_BOOK_ID, how='left')
    user_pref_emb = user_book_emb.groupby(constants.COL_USER_ID)[book_emb_cols].mean().reset_index()

    # Merge to df
    df = df.merge(user_pref_emb, on=constants.COL_USER_ID, how='left', suffixes=('', '_user_pref'))

    # Cosine sim for book
    user_pref_matrix = df[[c + '_user_pref' for c in book_emb_cols]].fillna(0).values
    book_matrix = df[book_emb_cols].fillna(0).values
    sim_book = 1 - cdist(user_pref_matrix, book_matrix, metric='cosine').diagonal()
    df['user_book_emb_sim'] = np.nan_to_num(sim_book, nan=0.0)

    # Clean up
    df.drop(columns=[c + '_user_pref' for c in book_emb_cols], inplace=True)
    return df


def add_target_encoding_and_interactions(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    print("Adding Target Encoding + Interactions + Rank features...")

    global_mean = train_df[config.TARGET].mean()

    # Smoothed Target Encoding
    # User
    user_stats = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(['mean', 'count'])
    user_stats['user_te'] = (user_stats['mean'] * user_stats['count'] + global_mean * 20) / (user_stats['count'] + 20)
    user_stats['user_ratings_count'] = user_stats['count']

    # Book
    book_stats = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(['mean', 'count'])
    book_stats['book_te'] = (book_stats['mean'] * book_stats['count'] + global_mean * 10) / (book_stats['count'] + 10)
    book_stats['book_ratings_count'] = book_stats['count']

    # Author
    author_stats = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(['mean', 'count'])
    author_stats['author_te'] = (author_stats['mean'] * author_stats['count'] + global_mean * 5) / (author_stats['count'] + 5)

    # Merge
    df = df.merge(user_stats[['user_te', 'user_ratings_count']], on=constants.COL_USER_ID, how='left')
    df = df.merge(book_stats[['book_te', 'book_ratings_count']], on=constants.COL_BOOK_ID, how='left')
    df = df.merge(author_stats[['author_te']], on=constants.COL_AUTHOR_ID, how='left')

    # Fill missing
    df['user_te'] = df['user_te'].fillna(global_mean)
    df['book_te'] = df['book_te'].fillna(global_mean)
    df['author_te'] = df['author_te'].fillna(global_mean)
    df['user_ratings_count'] = df['user_ratings_count'].fillna(0)
    df['book_ratings_count'] = df['book_ratings_count'].fillna(0)

    # Interaction Features
    df['user_book_te_mult'] = df['user_te'] * df['book_te']
    df['user_book_te_diff'] = df['user_te'] - df['book_te']
    df['user_book_count_ratio'] = (df['user_ratings_count'] + 1) / (df['book_ratings_count'] + 1)
    df['user_author_te'] = df['user_te'] * df['author_te']

    # Rank Features
    train_sorted = train_df.sort_values(constants.COL_TIMESTAMP)
    train_sorted['user_rating_order'] = train_sorted.groupby(constants.COL_USER_ID).cumcount() + 1
    train_sorted['user_total_ratings'] = train_sorted.groupby(constants.COL_USER_ID)[config.TARGET].transform('count')
    train_sorted['user_rating_pct'] = train_sorted['user_rating_order'] / train_sorted['user_total_ratings']

    # Merge rank features
    rank_cols = ['user_rating_pct']
    df = df.merge(
        train_sorted[[constants.COL_USER_ID, constants.COL_BOOK_ID] + rank_cols],
        on=[constants.COL_USER_ID, constants.COL_BOOK_ID],
        how='left'
    )
    df['user_rating_pct'] = df['user_rating_pct'].fillna(0.5)  # среднее положение

    # New additions
    df = add_temporal_features(df)
    df = add_similarity_features(df, train_df)

    # New: User age - publication year diff
    df['user_book_age_diff'] = df[constants.COL_AGE] - df[constants.COL_PUBLICATION_YEAR]

    # New: Gender-book interaction (mean rating per gender)
    gender_mean = train_df.groupby(constants.COL_GENDER)[config.TARGET].mean()
    df['gender_mean_rating'] = df[constants.COL_GENDER].map(gender_mean).fillna(global_mean)

    print("Target Encoding + Interactions + Rank + New features added!")
    return df


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values using training data statistics to avoid leakage.

    Args:
        df (pd.DataFrame): The main DataFrame to impute.
        train_df (pd.DataFrame): The training portion for fill values.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Categorical features: fill with 'missing'
    cat_features = config.CAT_FEATURES
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna(constants.MISSING_CAT_VALUE)

    # Numerical features: fill with median from train
    num_features = df.select_dtypes(include=["int", "float"]).columns.tolist()
    num_features = [col for col in num_features if col not in [config.TARGET, constants.COL_PREDICTION]]

    for col in num_features:
        if col in df.columns:
            fill_value = train_df[col].median() if col in train_df.columns else constants.MISSING_NUM_VALUE
            df[col] = df[col].fillna(fill_value)

    # Clip aggregates to [0,10]
    for col in [constants.F_USER_MEAN_RATING, constants.F_BOOK_MEAN_RATING, constants.F_AUTHOR_MEAN_RATING]:
        if col in df.columns:
            df[col] = df[col].clip(0, 10)

    print("Missing values handled.")
    return df


def feature_selection(model, features, threshold=0.01) -> list:
    """Select features based on CatBoost importance."""
    importances = model.get_feature_importance()
    imp_df = pd.DataFrame({'feature': features, 'importance': importances})
    selected = imp_df[imp_df['importance'] > threshold]['feature'].tolist()
    print(f"Selected {len(selected)} features out of {len(features)} (threshold={threshold})")
    return selected


def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    """Main feature creation pipeline.

    Args:
        df (pd.DataFrame): Merged DataFrame.
        book_genres_df (pd.DataFrame): Book genres DataFrame.
        descriptions_df (pd.DataFrame): Book descriptions DataFrame.
        include_aggregates (bool): Whether to include aggregate features.

    Returns:
        pd.DataFrame: DataFrame with all features added.
    """
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = add_book_and_author_embeddings(df, train_df, descriptions_df)
    df = add_target_encoding_and_interactions(df, train_df)
    df = handle_missing_values(df, train_df)

    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
