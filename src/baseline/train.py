"""
Main training script for the LightGBM model.

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor, Pool
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, constants
from .features import add_aggregate_features, handle_missing_values, add_target_encoding_and_interactions
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def train() -> None:
    """Runs the model training pipeline with temporal split.

    Loads prepared data from data/processed/, performs temporal split based on
    absolute date threshold, computes aggregate features on train split only,
    and trains a single LightGBM model. This ensures methodologically correct
    validation without data leakage from future timestamps.

    Note: Data must be prepared first using prepare_data.py
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train and test sets
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Check for timestamp column
    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set. "
            "Make sure data was prepared with timestamp preserved."
        )

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Perform temporal split
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    # Split data
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Verify temporal correctness
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"Max train timestamp: {max_train_timestamp}")
    print(f"Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError(
            f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
            f"is not greater than max train timestamp ({max_train_timestamp})."
        )
    print("✅ Temporal split validation passed: all validation timestamps are after train timestamps")

    # ← ВСТАВЬ ЭТУ СТРОКУ ВМЕСТО add_aggregate_features
    print("\nComputing advanced features on train split...")
    train_split_final = add_target_encoding_and_interactions(train_split.copy(), train_split)
    val_split_final = add_target_encoding_and_interactions(val_split.copy(), train_split)

    # Handle missing values (остаётся)
    train_split_final = handle_missing_values(train_split_final, train_split)
    val_split_final = handle_missing_values(val_split_final, train_split)

    # Features
    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [col for col in train_split_final.columns if col not in exclude_cols]
    features = [f for f in features if train_split_final[f].dtype != 'object']

    X_train = train_split_final[features]
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features]
    y_val = val_split_final[config.TARGET]

    # ← CATBOOST!
    cat_features = [col for col in config.CAT_FEATURES if col in features]

    print(f"\nTraining CatBoost on {len(features)} features ({len(cat_features)} categorical)...")
    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=200,
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        devices='0' if torch.cuda.is_available() else None
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model.fit(train_pool, eval_set=val_pool)

    # Evaluation
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    print(f"\nValidation RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Save
    model_path = config.MODEL_DIR / "catboost_model.cbm"
    model.save_model(str(model_path))
    print(f"CatBoost model saved to {model_path}")


if __name__ == "__main__":
    train()
