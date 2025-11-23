"""
Main training script for the CatBoost model (optimized for rating prediction).
"""

import catboost as cb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def train() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}")

    print(f"Loading data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows, {len(featured_df.columns)} columns")

    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(f"Column {constants.COL_TIMESTAMP} not found!")

    train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Temporal split
    print(f"\nTemporal split (ratio {config.TEMPORAL_SPLIT_RATIO})...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date → {split_date.date()}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)
    train_split = train_set[train_mask].copy()
    val_split   = train_set[val_mask].copy()

    print(f"Train: {len(train_split):,} | Val: {len(val_split):,}")

    # Добавляем агрегаты и сразу удаляем служебные колонки (time_decay и др.)
    print("\nAdding aggregate features...")
    train_split = add_aggregate_features(train_split, train_split)
    val_split   = add_aggregate_features(val_split,   train_split)

    # Удаляем все временные колонки, которые могли появиться
    temp_cols = [c for c in train_split.columns if c.startswith("time_decay") or c == "time_decay"]
    train_split = train_split.drop(columns=temp_cols, errors="ignore")
    val_split   = val_split.drop(columns=temp_cols, errors="ignore")

    # Обработка пропусков
    print("Handling missing values...")
    train_split = handle_missing_values(train_split, train_split)
    val_split   = handle_missing_values(val_split,   train_split)

    # Список фичей
    exclude_cols = {
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    }
    features = [c for c in train_split.columns if c not in exclude_cols]
    features = [c for c in features if train_split[c].dtype.name != "object"]

    cat_features = [c for c in features if train_split[c].dtype.name == "category"]

    X_train, y_train = train_split[features], train_split[config.TARGET]
    X_val,   y_val   = val_split[features],   val_split[config.TARGET]

    print(f"Features: {len(features)} (categorical: {len(cat_features)})")

    # Pools
    train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
    val_pool   = cb.Pool(X_val,   y_val,   cat_features=cat_features)

    # Оптимальные параметры + метрика RMSEWithUncertainty
    model = cb.CatBoostRegressor(
        iterations=10000,
        learning_rate=0.03,
        depth=9,
        l2_leaf_reg=5.0,
        bagging_temperature=0.9,
        random_strength=1.0,
        border_count=254,
        grow_policy="Lossguide",
        min_data_in_leaf=5,
        loss_function="RMSEWithUncertainty",   # Лучшая метрика для рейтингов
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=config.EARLY_STOPPING_ROUNDS,
        random_seed=config.RANDOM_STATE,
        verbose=200,
        thread_count=-1,
        # task_type="GPU", devices="0",  # раскомментируй, если есть GPU
    )

    print("\nStarting training with RMSEWithUncertainty...")
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False,
    )

    # Оценка
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae  = mean_absolute_error(y_val, val_pred)

    print(f"\nValidation RMSE: {rmse:.5f}")
    print(f"Validation MAE : {mae:.5f}")

    # Сохранение
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / "catboost_model.cbm"
    model.save_model(str(model_path))
    print(f"Model saved → {model_path}")

    joblib.dump(features, config.MODEL_DIR / "feature_columns.pkl")
    print("Feature list saved")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train()
