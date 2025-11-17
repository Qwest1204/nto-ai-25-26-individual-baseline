"""
Inference script to generate predictions for the test set.

Computes aggregate features on all train data and applies them to test set,
then generates predictions using the trained model.
"""

import numpy as np
import pandas as pd

from . import config, constants
from .features import handle_missing_values, add_target_encoding_and_interactions


def predict() -> None:
    """Generates and saves predictions for the test set.

    This script loads prepared data from data/processed/, computes aggregate features
    on all train data, applies them to test set, and generates predictions using
    the trained model.

    Note: Data must be prepared first using prepare_data.py, and model must be trained
    using train.py
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
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    print(f"Train set: {len(train_set):,} rows")
    print(f"Test set: {len(test_set):,} rows")

    # Compute advanced features on all train data (to use for test predictions)
    print("\nComputing advanced features on all train data...")
    test_set_final = add_target_encoding_and_interactions(test_set.copy(), train_set)

    # Handle missing values (use train_set for fill values)
    print("Handling missing values...")
    test_set_final = handle_missing_values(test_set_final, train_set)

    # Define features (exclude source, target, prediction, timestamp columns)
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in test_set_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = test_set_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_test = test_set_final[features]
    print(f"Prediction features: {len(features)}")

    # Load trained model
    model_path = config.MODEL_DIR / "catboost_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. " "Please run 'poetry run python -m src.baseline.train' first."
        )

    print(f"\nLoading model from {model_path}...")
    from catboost import CatBoostRegressor, Pool
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    cat_features = [col for col in config.CAT_FEATURES if col in features]
    for col in cat_features:
        X_test[col] = X_test[col].astype(str)

    test_pool = Pool(X_test, cat_features=cat_features)
    test_preds = model.predict(test_pool)
    ft_path = config.MODEL_DIR / 'ft_transformer.pt'
    if ft_path.exists():
        print("Loading FT-Transformer for ensemble...")
        ft_state = torch.load(ft_path)
        cat_feats = [col for col in config.CAT_FEATURES if col in X_test.columns]
        num_feats = [col for col in X_test.columns if col not in cat_feats]

        X_test_num, X_test_cat, _, _, _ = prepare_data_for_nn(test_set_final, cat_feats, num_feats, fit=False,
                                                              encoders=ft_state['encoders'], scaler=ft_state['scaler'])

        test_ds = RatingDataset(X_test_num, X_test_cat)
        test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

        ft_model = FTTransformer(len(num_feats), [len(ft_state['encoders'][c].classes_) for c in cat_feats])
        ft_model.load_state_dict(ft_state['model'])
        ft_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ft_model.to(device)

        ft_preds = []
        with torch.no_grad():
            for batch in test_loader:
                x_num, x_cat = [b.to(device) for b in batch]
                ft_preds.append(ft_model(x_num, x_cat).cpu().numpy())
        ft_preds = np.concatenate(ft_preds)

        test_preds = 0.6 * test_preds + 0.4 * ft_preds  # Ансамбль!
    # Clip predictions to be within the valid rating range [0, 10]
    clipped_preds = np.clip(test_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    # Create submission file
    submission_df = test_set[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission_df[constants.COL_PREDICTION] = clipped_preds

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file created at: {submission_path}")
    print(f"Predictions: min={clipped_preds.min():.4f}, max={clipped_preds.max():.4f}, mean={clipped_preds.mean():.4f}")


if __name__ == "__main__":
    predict()
