# train.py (полный код с улучшениями)

# src/baseline/train.py (updated — replace your file with this)

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
import torch  # Added import for torch.cuda.is_available()
from torch.utils.data import DataLoader

from . import config, constants
from .features import add_target_encoding_and_interactions, handle_missing_values, feature_selection
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date
from .nn_model import train_ft_transformer, FTTransformer, prepare_data_for_nn, RatingDataset


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

    print("\nComputing advanced features on train split...")
    train_split_final = add_target_encoding_and_interactions(train_split.copy(), train_split)
    val_split_final = add_target_encoding_and_interactions(val_split.copy(), train_split)

    # Handle missing values
    train_split_final = handle_missing_values(train_split_final, train_split)
    val_split_final = handle_missing_values(val_split_final, train_split)

    # Features
    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [col for col in train_split_final.columns if col not in exclude_cols]
    features = [f for f in features if train_split_final[f].dtype != 'object']

    X_train = train_split_final[features]
    y_train = train_split_final[config.TARGET].clip(0, 10)
    X_val = val_split_final[features]
    y_val = val_split_final[config.TARGET].clip(0, 10)

    cat_features = [col for col in config.CAT_FEATURES if col in features]

    # Convert categorical features to string to fix CatBoost type error
    for col in cat_features:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    print(f"\nTraining CatBoost on {len(features)} features ({len(cat_features)} categorical)...")
    model = CatBoostRegressor(
        iterations=7000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=300,
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        devices='0' if torch.cuda.is_available() else None
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model.fit(train_pool, eval_set=val_pool)

    # Feature selection after CatBoost
    selected_features = feature_selection(model, features)
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    cat_features = [col for col in cat_features if col in selected_features]

    # Train LightGBM
    print("\nTraining LightGBM...")
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=cat_features)
    lgb_model = lgb.train(
        config.LGB_PARAMS,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=300)],
    )

    print("\nTraining FT-Transformer...")
    cat_feats = [col for col in config.CAT_FEATURES if col in selected_features]
    num_feats = [col for col in selected_features if col not in cat_feats]

    train_ft_transformer(train_split_final[selected_features], train_split_final[config.TARGET],
                         val_split_final[selected_features], val_split_final[config.TARGET],
                         cat_feats, num_feats)

    # === ENSEMBLE EVALUATION ON VALIDATION ===
    print("\nEvaluating ensemble on validation split...")
    cat_val_preds = model.predict(X_val)
    lgb_val_preds = lgb_model.predict(X_val)

    ft_state = torch.load(config.MODEL_DIR / 'ft_transformer.pt', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ft_model = FTTransformer(len(num_feats), [len(ft_state['encoders'][c].classes_) for c in cat_feats]).to(device)
    ft_model.load_state_dict(ft_state['model'])
    ft_model.eval()

    X_val_num, X_val_cat, _, _, _ = prepare_data_for_nn(val_split_final, cat_feats, num_feats, fit=False,
                                                        encoders=ft_state['encoders'], scaler=ft_state['scaler'])
    val_ds = RatingDataset(X_val_num, X_val_cat)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

    ft_val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            x_num, x_cat = [b.to(device) for b in batch]
            ft_val_preds.append(ft_model(x_num, x_cat).cpu().numpy())
    ft_val_preds = np.concatenate(ft_val_preds)

    ensemble_val_preds = 0.5 * cat_val_preds + 0.3 * ft_val_preds + 0.2 * lgb_val_preds
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
    print(f"\nEnsemble Val RMSE: {ensemble_rmse:.4f}")

    # Evaluation (CatBoost only for reference)
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    print(f"\nCatBoost Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Save models
    model_path = config.MODEL_DIR / "catboost_model.cbm"
    model.save_model(str(model_path))
    print(f"CatBoost model saved to {model_path}")

    lgb_model.save_model(str(config.MODEL_DIR / "lightgbm_model.txt"))
    print(f"LightGBM model saved to {config.MODEL_DIR / 'lightgbm_model.txt'}")

    # FT saved in train_ft_transformer


if __name__ == "__main__":
    train()
