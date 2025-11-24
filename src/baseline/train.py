"""
train.py — финальная версия с XGBoost вместо CatBoost
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def train() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    print("Loading processed data...")
    df = pd.read_parquet(processed_path, engine="pyarrow")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Temporal split
    print(f"Temporal split (ratio {config.TEMPORAL_SPLIT_RATIO})")
    split_date = get_split_date_from_ratio(train_df, config.TEMPORAL_SPLIT_RATIO)
    train_mask, val_mask = temporal_split_by_date(train_df, split_date)

    train_split = train_df[train_mask].copy()
    val_split   = train_df[val_mask].copy()

    print(f"Train rows: {len(train_split):,} | Val rows: {len(val_split):,}")

    # Aggregate features (leak-free)
    train_split = add_aggregate_features(train_split, train_split)
    val_split   = add_aggregate_features(val_split,   train_split)

    # Удаляем служебные колонки
    train_split = train_split.drop(columns=["time_decay"], errors="ignore")
    val_split   = val_split.drop(columns=["time_decay"], errors="ignore")

    # Пропуски
    train_split = handle_missing_values(train_split, train_split)
    val_split   = handle_missing_values(val_split,   train_split)

    # Формируем список признаков
    exclude = {constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP}
    features = [c for c in train_split.columns if c not in exclude and train_split[c].dtype.name != "object"]
    cat_features = [c for c in features if train_split[c].dtype.name == "category"]

    # XGBoost требует, чтобы категориальные признаки были преобразованы в int коды
    for col in cat_features:
        train_split[col] = train_split[col].cat.codes.astype("category")
        val_split[col]   = val_split[col].cat.codes.astype("category")

    X_train, y_train = train_split[features], train_split[config.TARGET]
    X_val,   y_val   = val_split[features],   val_split[config.TARGET]

    print(f"Using {len(features)} features ({len(cat_features)} categorical → converted to codes)")

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)

    params = {
        "objective":          "reg:squarederror",
        "eval_metric":        "rmse",
        "tree_method":        "hist",          # быстро и поддерживает categorical
        "learning_rate":      0.03,
        "max_depth":          9,
        "subsample":          0.9,
        "colsample_bytree":   0.9,
        "reg_lambda":         5.0,
        "reg_alpha":          0.0,
        "min_child_weight":   5,
        "seed":               config.RANDOM_STATE,
        "enable_categorical": True,
    }

    print("\nTraining XGBoost...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=20000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose_eval=200,
    )

    best_iter = model.best_iteration
    val_pred = model.predict(dval)

    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae  = mean_absolute_error(y_val, val_pred)
    print(f"\nBest iteration: {best_iter}")
    print(f"Validation RMSE: {rmse:.5f}")
    print(f"Validation MAE : {mae:.5f}")

    # Сохранение
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / "xgboost_model.json"
    model.save_model(model_path)
    joblib.dump(features, config.MODEL_DIR / "feature_columns.pkl")

    print(f"\nModel saved → {model_path}")
    print("Training completed!")


if __name__ == "__main__":
    train()
