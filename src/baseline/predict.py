import catboost as cb
import joblib
import numpy as np
import pandas as pd

from . import config, constants
from .features import add_aggregate_features, handle_missing_values


def predict() -> None:
    path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Нет обработанных данных: {path}")

    print(f"Грузим данные из {path}")
    df = pd.read_parquet(path)

    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_df  = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    # агрегации без лика
    test_df = add_aggregate_features(test_df, train_df)

    # мусорная колонка
    test_df.drop(columns=["time_decay"], errors="ignore", inplace=True)

    # заполняем пропуски как на трейне
    test_df = handle_missing_values(test_df, train_df)

    # пытаемся взять список фичей из обучения
    feat_path = config.MODEL_DIR / "feature_columns.pkl"
    if feat_path.exists():
        features = joblib.load(feat_path)
    else:
        print("Список фичей не найден — беру всё кроме служебных")
        exclude = {constants.COL_SOURCE, config.TARGET,
                   constants.COL_PREDICTION, constants.COL_TIMESTAMP}
        features = [c for c in test_df.columns
                    if c not in exclude and test_df[c].dtype.name != "object"]

    X_test = test_df[features]

    model_path = config.MODEL_DIR / "catboost_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model = cb.CatBoostRegressor()
    model.load_model(str(model_path))

    # предиктим
    preds = model.predict(X_test)
    if preds.ndim == 2 and preds.shape[1] == 2:
        preds = preds[:, 0]
    else:
        preds = preds.ravel()

    # клипаем в разрешённый диапазон
    preds = np.clip(preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    # формируем сабмит
    sub = test_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    sub[constants.COL_PREDICTION] = preds

    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME
    sub.to_csv(out_path, index=False)

    print(f"Готово → {out_path}")
    print(f"pred: min {preds.min():.4f} | max {preds.max():.4f} | mean {preds.mean():.4f}")


if __name__ == "__main__":
    predict()
