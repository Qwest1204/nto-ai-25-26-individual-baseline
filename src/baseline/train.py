"""
train.py — финальная версия
CatBoost + RMSEWithUncertainty + корректное сохранение
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import catboost as cb

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

    # Aggregate features (без утечек)
    train_split = add_aggregate_features(train_split, train_split)
    val_split   = add_aggregate_features(val_split,   train_split)

    # Убираем служебные колонки
    train_split = train_split.drop(columns=["time_decay"], errors="ignore")
    val_split   = val_split.drop(columns=["time_decay"], errors="ignore")

    # Пропуски
    train_split = handle_missing_values(train_split, train_split)
    val_split   = handle_missing_values(val_split,   train_split)

    # Фичи
    exclude = {constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP}
    features = [c for c in train_split.columns if c not in exclude and train_split[c].dtype.name != "object"]
    cat_features = [c for c in features if train_split[c].dtype.name == "category"]

    X_train, y_train = train_split[features], train_split[config.TARGET]
    X_val,   y_val   = val_split[features],   val_split[config.TARGET]

    print(f"Using {len(features)} features ({len(cat_features)} categorical)")

    train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
    val_pool   = cb.Pool(X_val,   y_val,   cat_features=cat_features)

    model = cb.CatBoostRegressor(
        iterations=10000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10.0,
        bagging_temperature=1.0,
        random_strength=1.0,
        border_count=254,
        grow_policy="Lossguide",
        min_data_in_leaf=5,
        loss_function="RMSEWithUncertainty",   # ← даёт лучший скор
        eval_metric="RMSE",
        #rsm=0.7,
        od_type="Iter",
        od_wait=config.EARLY_STOPPING_ROUNDS,
        random_seed=config.RANDOM_STATE,
        verbose=200,
        thread_count=-1,
        task_type="GPU", devices="0",  # раскомментируй при наличии GPU
    )

    print("\nTraining started...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Предсказание только mean (RMSEWithUncertainty выдаёт 2 столбца)
    val_pred = model.predict(X_val)
    if val_pred.ndim == 2:
        val_pred = val_pred[:, 0]

    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae  = mean_absolute_error(y_val, val_pred)
    print(f"\nValidation RMSE: {rmse:.5f}")
    print(f"Validation MAE : {mae:.5f}")

    # Сохранение
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.MODEL_DIR / "catboost_model.cbm"))
    joblib.dump(features, config.MODEL_DIR / "feature_columns.pkl")

    print(f"\nModel saved → {config.MODEL_DIR / 'catboost_model.cbm'}")
    print("Training completed!")


if __name__ == "__main__":
    train()
