# src/baseline/features.py
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from . import config, constants


# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    if constants.F_BOOK_GENRES_COUNT in df.columns:
        return df
    print("Adding genre count feature...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID).size().reset_index(name=constants.F_BOOK_GENRES_COUNT)
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    if any(c.startswith("tfidf_") for c in df.columns):
        print("TF-IDF already present, skipping")
        return df

    print("Adding TF-IDF features...")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    train_books = set(train_df[constants.COL_BOOK_ID])
    train_desc = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_desc[constants.COL_DESCRIPTION] = train_desc[constants.COL_DESCRIPTION].fillna("")

    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(**{k.lower(): v for k, v in config.__dict__.items() if k.startswith("TFIDF_")})
        vectorizer.fit(train_desc[constants.COL_DESCRIPTION])
        joblib.dump(vectorizer, vectorizer_path)

    all_desc = descriptions_df.set_index(constants.COL_BOOK_ID)[constants.COL_DESCRIPTION].fillna("")
    desc_series = df[constants.COL_BOOK_ID].map(all_desc).fillna("")

    tfidf_matrix = vectorizer.transform(desc_series)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])], index=df.index)
    return pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)


def add_bert_features(df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    if any(c.startswith("labse_") for c in df.columns):
        print("LaBSE embeddings already present, skipping")
        return df

    print("Adding LaBSE sentence embeddings...")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = config.MODEL_DIR / "labse_embeddings.pkl"

    if emb_path.exists():
        embeddings_dict = joblib.load(emb_path)
    else:
        model = SentenceTransformer('sentence-transformers/LaBSE')
        desc_dict = descriptions_df.set_index(constants.COL_BOOK_ID)[constants.COL_DESCRIPTION].fillna("").to_dict()
        book_ids = list(desc_dict.keys())
        texts = [desc_dict[bid] for bid in book_ids]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        embeddings_dict = dict(zip(book_ids, embeddings))
        joblib.dump(embeddings_dict, emb_path)

    zero_vec = np.zeros(768)
    vectors = df[constants.COL_BOOK_ID].map(lambda x: embeddings_dict.get(x, zero_vec))
    emb_array = np.stack(vectors.values)

    for i in range(emb_array.shape[1]):
        df[f"labse_{i}"] = emb_array[:, i]

    return df


# ====================== ОСНОВНАЯ ФУНКЦИЯ ======================
def create_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame | None = None,
    descriptions_df: pd.DataFrame | None = None,
    include_aggregates: bool = False,
) -> pd.DataFrame:
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    # Жанры
    if book_genres_df is not None:
        df = add_genre_features(df, book_genres_df)

    # Текстовые фичи — передаём descriptions_df только если он есть
    if descriptions_df is not None:
        df = add_text_features(df, train_df, descriptions_df)
        df = add_bert_features(df, descriptions_df)
    else:
        print("No descriptions_df provided — skipping text features (they should already be in df)")

    # Advanced фичи (если файл импортирован)
    try:
        from .features_advanced import add_advanced_features
        df = add_advanced_features(df, train_df)
    except ImportError:
        print("features_advanced.py not found — skipping advanced features")

    df = handle_missing_values(df, train_df)

    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"Feature engineering complete → {df.shape[1]} columns")
    return df
