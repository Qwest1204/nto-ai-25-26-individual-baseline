
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config, constants


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    print("Делаем безопасные агрегаты...")

    train_read = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    global_mean = train_read[config.TARGET].mean()

    # пользователь
    user_agg = train_read.groupby(constants.COL_USER_ID)[config.TARGET].agg(['mean', 'count'])
    user_agg.columns = [constants.F_USER_MEAN_RATING, constants.F_USER_RATINGS_COUNT]
    user_agg = user_agg.reset_index()

    # книга
    book_agg = train_read.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(['mean', 'count'])
    book_agg.columns = [constants.F_BOOK_MEAN_RATING, constants.F_BOOK_RATINGS_COUNT]
    book_agg = book_agg.reset_index()

    # автор (если найдём books.csv)
    author_agg = None
    books_path = None
    for p in [config.DATA_DIR / "raw" / constants.BOOK_DATA_FILENAME,
              config.DATA_DIR / constants.BOOK_DATA_FILENAME]:
        if p.exists():
            books_path = p
            break

    if books_path:
        try:
            books_df = pd.read_csv(books_path)
            if constants.COL_AUTHOR_ID in books_df.columns:
                tmp = train_read.merge(books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
                                       on=constants.COL_BOOK_ID, how='left')
                tmp = tmp.dropna(subset=[constants.COL_AUTHOR_ID])
                if not tmp.empty:
                    author_agg = tmp.groupby(constants.COL_AUTHOR_ID)[config.TARGET].mean().reset_index()
                    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]
        except Exception as e:
            print("Не получилось загрузить авторов:", e)

    # мержим всё
    df = df.merge(user_agg, on=constants.COL_USER_ID, how='left')
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how='left')

    if author_agg is not None:
        if books_path:
            books_df = pd.read_csv(books_path)  # перечитываем если нужно
            df = df.merge(books_df[[constants.COL_BOOK_ID, constants.COL_AUTHOR_ID]],
                          on=constants.COL_BOOK_ID, how='left')
        df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how='left')

    # заполняем холодных
    df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean * 1.05)
    df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    # простые взаимодействия
    df['user_book_diff'] = df[constants.F_USER_MEAN_RATING] - df[constants.F_BOOK_MEAN_RATING]
    df['user_book_abs_diff'] = df['user_book_diff'].abs()
    df['user_book_compatibility'] = 10 - df['user_book_abs_diff']

    df['user_pop_log'] = np.log1p(df[constants.F_USER_RATINGS_COUNT])
    df['book_pop_log'] = np.log1p(df[constants.F_BOOK_RATINGS_COUNT])
    df['pop_product'] = df['user_pop_log'] * df['book_pop_log']

    df['expected_rating'] = df[constants.F_USER_MEAN_RATING] * 0.5 + df[constants.F_BOOK_MEAN_RATING] * 0.5
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df['expected_rating'] = df['expected_rating'] * 0.8 + df[constants.F_AUTHOR_MEAN_RATING] * 0.2
    df['expected_rating'] = df['expected_rating'].clip(0, 10)

    print("Агрегаты готовы")
    return df


def add_to_read_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    to_read = train_df[train_df[constants.COL_HAS_READ] == 0]
    cnt = to_read.groupby(constants.COL_USER_ID).size().reset_index(name='user_to_read_cnt')
    return df.merge(cnt, on=constants.COL_USER_ID, how='left').fillna({'user_to_read_cnt': 0})


def add_cf_embeddings(df: pd.DataFrame, train_df: pd.DataFrame, n_comp=50) -> pd.DataFrame:
    print("SVD эмбеддинги...")
    ratings = train_df[train_df[constants.COL_HAS_READ] == 1]
    pivot = ratings.pivot_table(index=constants.COL_USER_ID, columns=constants.COL_BOOK_ID,
                                values=config.TARGET, fill_value=0)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    user_emb = svd.fit_transform(pivot)
    book_emb = svd.components_.T

    user_df = pd.DataFrame(user_emb, index=pivot.index,
                           columns=[f'user_svd_{i}' for i in range(n_comp)])
    book_df = pd.DataFrame(book_emb, index=pivot.columns,
                           columns=[f'book_svd_{i}' for i in range(n_comp)])

    df = df.merge(user_df, on=constants.COL_USER_ID, how='left')
    df = df.merge(book_df, on=constants.COL_BOOK_ID, how='left')
    return df.fillna(0)


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    # сколько жанров у книги
    cnt = book_genres_df.groupby(constants.COL_BOOK_ID).size().reset_index(name=constants.F_BOOK_GENRES_COUNT)
    df = df.merge(cnt, on=constants.COL_BOOK_ID, how='left')

    # one-hot топ жанров
    grouped = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].apply(list).reset_index()
    mlb = MultiLabelBinarizer(sparse_output=True)
    onehot = mlb.fit_transform(grouped[constants.COL_GENRE_ID])
    onehot_df = pd.DataFrame.sparse.from_spmatrix(onehot,
                                                  index=grouped.index,
                                                  columns=[f'genre_{c}' for c in mlb.classes_])
    onehot_df[constants.COL_BOOK_ID] = grouped[constants.COL_BOOK_ID]
    onehot_df = onehot_df.sparse.to_dense().fillna(0)

    df = df.merge(onehot_df, on=constants.COL_BOOK_ID, how='left')
    return df


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    print("TF-IDF из описаний...")
    vec_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_books = train_df[constants.COL_BOOK_ID].unique()
    train_desc = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_desc[constants.COL_DESCRIPTION] = train_desc[constants.COL_DESCRIPTION].fillna("")

    if vec_path.exists():
        vec = joblib.load(vec_path)
    else:
        vec = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES,
                              min_df=config.TFIDF_MIN_DF,
                              max_df=config.TFIDF_MAX_DF,
                              ngram_range=config.TFIDF_NGRAM_RANGE)
        vec.fit(train_desc[constants.COL_DESCRIPTION])
        joblib.dump(vec, vec_path)

    desc_map = dict(zip(descriptions_df[constants.COL_BOOK_ID],
                        descriptions_df[constants.COL_DESCRIPTION].fillna("")))

    texts = df[constants.COL_BOOK_ID].map(desc_map).fillna("")
    tfidf_mat = vec.transform(texts)
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_mat,
                                                 columns=[f'tfidf_{i}' for i in range(tfidf_mat.shape[1])])
    tfidf_df.index = df.index

    df = pd.concat([df, tfidf_df], axis=1)
    return df


def add_bert_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    print("BERT эмбеддинги...")
    emb_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
    model.eval()
    model.to(config.BERT_DEVICE)

    desc_map = descriptions_df.set_index(constants.COL_BOOK_ID)[constants.COL_DESCRIPTION].fillna("").to_dict()

    if emb_path.exists():
        emb_dict = joblib.load(emb_path)
    else:
        emb_dict = {}
        books = df[constants.COL_BOOK_ID].unique()
        for book_id in tqdm(books, desc="BERT"):
            text = desc_map.get(book_id, "")
            if not text:
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=config.BERT_MAX_LENGTH, padding=True)
            inputs = {k: v.to(config.BERT_DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                emb = model(**inputs).last_hidden_state.mean(1).cpu().numpy()[0]
            emb_dict[book_id] = emb
        joblib.dump(emb_dict, emb_path)

    vectors = [emb_dict.get(bid, np.zeros(config.BERT_EMBEDDING_DIM)) for bid in df[constants.COL_BOOK_ID]]
    bert_df = pd.DataFrame(vectors, columns=[f'bert_{i}' for i in range(config.BERT_EMBEDDING_DIM)], index=df.index)
    return pd.concat([df, bert_df], axis=1)


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    print("Заполняем пропуски...")
    global_mean = train_df[config.TARGET].mean()

    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(df[constants.COL_AGE].median())

    fill_cols = [
        constants.F_USER_MEAN_RATING, constants.F_BOOK_MEAN_RATING,
        constants.F_AUTHOR_MEAN_RATING, constants.COL_AVG_RATING
    ]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(global_mean)

    cnt_cols = [constants.F_USER_RATINGS_COUNT, constants.F_BOOK_RATINGS_COUNT, constants.F_BOOK_GENRES_COUNT]
    for c in cnt_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # нули для всех эмбеддингов и one-hot
    for prefix in ['tfidf_', 'bert_', 'user_svd_', 'book_svd_', 'genre_']:
        cols = [c for c in df.columns if c.startswith(prefix)]
        df[cols] = df[cols].fillna(0)

    # категориальные
    for col in config.CAT_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].astype(str).fillna('missing').astype('category')

    return df


def create_features(df: pd.DataFrame, book_genres_df: pd.DataFrame,
                    descriptions_df: pd.DataFrame, include_aggregates=False) -> pd.DataFrame:
    print("Запуск генерации фичей...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_to_read_features(df, train_df)
    df = add_cf_embeddings(df, train_df)
    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = add_bert_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    # категории для катбустa
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Все фичи готовы")
    return df
