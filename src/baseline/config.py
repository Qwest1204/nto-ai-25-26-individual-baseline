"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# --- PARAMETERS ---
N_SPLITS = 5  # Deprecated
RANDOM_STATE = 42
TARGET = constants.COL_TARGET

TEMPORAL_SPLIT_RATIO = 0.95

EARLY_STOPPING_ROUNDS = 200
MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt"  # Deprecated
MODEL_FILENAME = "lgb_model.txt"
NOMIC_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'

TFIDF_MAX_FEATURES = 1000  # Увеличил с 500 для больше фич
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 3)  # Добавил tri-grams для capture phrases

NOMIC_BATCH_SIZE = 16
NOMIC_MAX_LENGTH = 1024
NOMIC_EMBEDDING_DIM = 256  # Оставил, но ниже используем 384 для FT
NOMIC_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
NOMIC_GPU_MEMORY_FRACTION = 0.75

CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
    'day_of_week',  # New: from timestamp
    'month',        # New
]

LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 5000,
    "learning_rate": 0.02,  # Уменьшил с 0.01 для slower learning
    "feature_fraction": 0.7,  # С 0.8, для regularization
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 50,  # Увеличил с 31 для complexity
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
}

LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
    "callbacks": [],
}
