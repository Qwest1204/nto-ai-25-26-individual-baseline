"""
Inference script — полностью совместим с моделью, обученной на RMSEWithUncertainty.
"""

import catboost as cb
import joblib
import numpy as np
import pandas as pd

from . import config, constants
from .features import add_aggregate_features, handle_missing_values


def predict() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}")

    print(f"Loading processed data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_set  = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    print(f"Train: {len(train_set):,} | Test: {len(test_set):,}")

    # Aggregate features (leak-free)
    print("\nAdding aggregate features to test...")
    #test_set = add_aggregate_features(test_set, train_set)

    # Удаляем временные колонки, если остались
    test_set = test_set.drop(columns=["time_decay"], errors="ignore")

    # Missing values
    print("Handling missing values...")
    test_set = handle_missing_values(test_set, train_set)

    # Загружаем список фичей, использованных при обучении
    features_path = config.MODEL_DIR / "feature_columns.pkl"
    if features_path.exists():
        features = joblib.load(features_path)
        print(f"Loaded {len(features)} features loaded from training")
    else:
        # fallback — все кроме служебных
        exclude = {constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP}
        features = [c for c in test_set.columns if c not in exclude and test_set[c].dtype.name != "object"]
        print(f"Warning: feature list not found — using auto-detected {len(features)} features")

    X_test = test_set[features]

    # Загрузка модели
    model_path = config.MODEL_DIR / "catboost_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = cb.CatBoostRegressor()
    model.load_model(str(model_path))

    # КЛЮЧЕВОЕ: берём только первый столбец (среднее значение)
    print("Predicting (taking mean from RMSEWithUncertainty)...")
    raw_preds = model.predict(X_test)          # может быть (n, 2) или (n,)

    if raw_preds.ndim == 2 and raw_preds.shape[1] == 2:
        test_preds = raw_preds[:, 0]           # ← берём только mean
        print(f"Detected RMSEWithUncertainty output → extracted mean (variance discarded)")
    else:
        test_preds = raw_preds.ravel()

    # Клиппинг в [0, 10]
    clipped_preds = np.clip(test_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    # Submission
    submission_df = test_set[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission_df[constants.COL_PREDICTION] = clipped_preds

    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission saved → {submission_path}")
    print(f"Predictions: min={clipped_preds.min():.4f}, max={clipped_preds.max():.4f}, mean={clipped_preds.mean():.4f}")


if __name__ == "__main__":
    predict()
