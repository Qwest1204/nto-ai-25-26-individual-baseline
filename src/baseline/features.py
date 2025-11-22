# src/baseline/features.py

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

from . import config, constants


# -------------------------- УТИЛИТЫ --------------------------
def _load_nomic_model():
    model = SentenceTransformer(
        config.NOMIC_MODEL_NAME,
        trust_remote_code=True,
        device=config.NOMIC_DEVICE
    )
    model.max_seq_length = config.NOMIC_MAX_LENGTH
    return model


# -------------------------- ОСНОВНЫЕ ФУНКЦИИ --------------------------
def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    print("Adding TF-IDF + Nomic embeddings...")
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # TF-IDF
    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        train_texts = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_df[constants.COL_BOOK_ID])][constants.COL_DESCRIPTION].fillna("")
        vectorizer.fit(train_texts)
        joblib.dump(vectorizer, vectorizer_path)

    all_texts = descriptions_df[constants.COL_DESCRIPTION].fillna("")
    tfidf_matrix = vectorizer.transform(all_texts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    tfidf_df[constants.COL_BOOK_ID] = descriptions_df[constants.COL_BOOK_ID].values
    df = df.merge(tfidf_df, on=constants.COL_BOOK_ID, how="left")

    # Nomic embeddings (384 dim)
    emb_path = config.MODEL_DIR / "nomic_book_embeddings_384.pkl"
    if emb_path.exists():
        book_emb_dict = joblib.load(emb_path)
    else:
        model = _load_nomic_model()
        texts = descriptions_df[constants.COL_DESCRIPTION].fillna("Нет описания").tolist()
        embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Nomic embeddings"):
            batch = texts[i:i + batch_size]
            embeddings.extend(model.encode(batch, show_progress_bar=False))
        embeddings = np.array(embeddings)
        book_emb_dict = dict(zip(descriptions_df[constants.COL_BOOK_ID], embeddings))
        joblib.dump(book_emb_dict, emb_path)

    # === ОДНИМ КОНКАТЕНАТОМ — УБИРАЕМ ФРАГМЕНТАЦИЮ ===
    emb_matrix = np.stack([book_emb_dict.get(bid, np.zeros(384)) for bid in df[constants.COL_BOOK_ID]])
    emb_cols = [f"nomic_book_{i}" for i in range(384)]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
    df = pd.concat([df, emb_df], axis=1)

    print(f"Added TF-IDF ({tfidf_matrix.shape[1]}) + Nomic (384) features")
    return df


def add_target_encoding_and_interactions(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    print("Adding target encoding + interactions...")
    global_mean = train_df[config.TARGET].mean()

    # Target encoding
    for col, prior in [("user_id", 20), ("book_id", 10), ("author_id", 5)]:
        if col not in train_df.columns:
            continue
        agg = train_df.groupby(col)[config.TARGET].agg(['mean', 'count'])
        te = (agg['mean'] * agg['count'] + global_mean * prior) / (agg['count'] + prior)
        df = df.merge(te.rename(f"{col}_te"), left_on=col, right_index=True, how="left")
        df[f"{col}_te"] = df[f"{col}_te"].fillna(global_mean)

    # Interactions
    df["user_book_te_diff"] = df["user_id_te"] - df["book_id_te"]
    df["user_book_te_mult"] = df["user_id_te"] * df["book_id_te"]

    # Rank in user history
    train_sorted = train_df.sort_values(constants.COL_TIMESTAMP)
    train_sorted["rank"] = train_sorted.groupby("user_id").cumcount() + 1
    train_sorted["total"] = train_sorted.groupby("user_id")["book_id"].transform("count")
    train_sorted["user_rating_pct"] = train_sorted["rank"] / train_sorted["total"]
    df = df.merge(train_sorted[["user_id", "book_id", "user_rating_pct"]], on=["user_id", "book_id"], how="left")
    df["user_rating_pct"] = df["user_rating_pct"].fillna(0.5)

    # Temporal features
    if constants.COL_TIMESTAMP in df.columns:
        df["day_of_week"] = pd.to_datetime(df[constants.COL_TIMESTAMP]).dt.dayofweek
        df["month"] = pd.to_datetime(df[constants.COL_TIMESTAMP]).dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    print("Target encoding + interactions added")
    return df


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Корректная обработка пропусков для категориальных колонок (без падения на Categorical)"""
    print("Handling missing values...")

    # 1. Категориальные колонки — добавляем 'missing' в категории, если нужно
    for col in config.CAT_FEATURES:
        if col not in df.columns:
            continue

        if pd.api.types.is_categorical_dtype(df[col]):
            # Если уже category — добавляем 'missing' в список категорий
            if 'missing' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('missing')
            df[col] = df[col].fillna('missing')
        else:
            # Если ещё не category — просто заполняем и потом сделаем category
            df[col] = df[col].fillna('missing')

    # 2. Числовые колонки — медиана из train
    num_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    num_cols = [c for c in num_cols if c not in [config.TARGET, "rating_predict"]]

    for col in num_cols:
        if col in train_df.columns:
            median_val = train_df[col].median()
        else:
            median_val = 0
        df[col] = df[col].fillna(median_val)

    # 3. Клиппинг target encoding фич (чтобы не вылезало за [0,10])
    te_cols = [c for c in df.columns if c.endswith("_te")]
    for col in te_cols:
        df[col] = df[col].clip(0, 10)

    # 4. Финальное приведение к category (уже безопасно)
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Missing values handled successfully!")
    return df


# -------------------------- ГЛАВНАЯ ФУНКЦИЯ --------------------------
def create_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = False
) -> pd.DataFrame:
    print("=== STARTING FEATURE ENGINEERING ===")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # 1. Текстовые признаки
    df = add_text_features(df, train_df, descriptions_df)

    # 2. Target encoding и взаимодействия
    df = add_target_encoding_and_interactions(df, train_df)

    # 3. Обработка пропусков — ОБЯЗАТЕЛЬНО!
    df = handle_missing_values(df, train_df)

    # 4. Категории
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"Feature engineering complete: {df.shape[1]} columns total")
    return df
