# src/baseline/predict.py
import joblib
import numpy as np
import pandas as pd

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .features_advanced import add_advanced_features


def predict() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError("Run prepare_data.py first!")

    df = pd.read_parquet(processed_path)
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    # Загружаем сохранённые объекты
    models = joblib.load(config.MODEL_DIR / "ensemble_models.pkl")
    features = joblib.load(config.MODEL_DIR / "features_list.pkl")
    weights = joblib.load(config.MODEL_DIR / "ensemble_weights.pkl")

    # Добавляем те же фичи, что и на train
    test_with_agg = add_aggregate_features(test_df, train_df)
    test_advanced = add_advanced_features(test_with_agg, train_df)
    test_final = handle_missing_values(test_advanced, train_df)

    X_test = test_final[features]

    # Предсказания всех моделей
    test_preds = np.zeros(len(X_test))
    for model, w in zip(models, weights):
        if isinstance(model, xgb.core.Booster):
            pred = model.predict(xgb.DMatrix(X_test))
        else:
            pred = model.predict(X_test)
        test_preds += w * pred

    test_preds = test_preds / sum(weights)

    # === ПОСТ-ПРОЦЕССИНГ ===
    # 1. Округление до 0.5
    test_preds = np.round(test_preds * 2) / 2

    # 2. Калибровка по среднему пользователя
    user_mean = train_df.groupby(constants.COL_USER_ID)[config.TARGET].mean()
    calib = test_final[constants.COL_USER_ID].map(user_mean).fillna(test_preds)
    test_preds = 0.7 * test_preds + 0.3 * calib

    test_preds = np.clip(test_preds, 0.5, 10)

    # Сабмит
    submission = test_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission[constants.COL_PREDICTION] = test_preds

    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.SUBMISSION_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"Submission saved: {out_path}")
    print(f"Preds: min={test_preds.min():.3f}, max={test_preds.max():.3f}, mean={test_preds.mean():.3f}")


if __name__ == "__main__":
    predict()
