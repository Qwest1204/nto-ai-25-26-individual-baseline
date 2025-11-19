# src/baseline/train.py
"""
Обновлённый тренинг с ансамблем: 3×LGB + XGBoost + CatBoost
+ все топовые фичи из features_advanced.py
"""

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, constants
from .features import create_features
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date
from .features_advanced import add_advanced_features  # ← НОВЫЕ ФИЧИ


def train() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Run prepare_data.py first! Missing {processed_path}")

    print("Loading processed data...")
    df = pd.read_parquet(processed_path, engine="pyarrow")

    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    if constants.COL_TIMESTAMP not in train_df.columns:
        raise ValueError("Timestamp column missing!")

    train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])

    # Temporal split
    split_date = get_split_date_from_ratio(train_df, config.TEMPORAL_SPLIT_RATIO)
    print(f"Temporal split date: {split_date}")
    train_mask, val_mask = temporal_split_by_date(train_df, split_date)

    train_split = train_df[train_mask].copy()
    val_split = train_df[val_mask].copy()

    print(f"Train: {len(train_split):,} | Val: {len(val_split):,}")

    # === ДОБАВЛЯЕМ ВСЕ ТОП-ФИЧИ ===
    print("Engineering features...")
    # Сначала базовые (включая LaBSE вместо старого BERT)
    full_featured = create_features(
        pd.concat([train_split, val_split], ignore_index=True),
        book_genres_df=None,  # уже в processed
        descriptions_df=None,
        include_aggregates=False
    )

    train_featured = full_featured.iloc[:len(train_split)].copy()
    val_featured = full_featured.iloc[len(train_split):].reset_index(drop=True)

    # Добавляем агрегаты и advanced фичи ТОЛЬКО на train_split
    train_featured = add_aggregate_features(train_featured, train_split)
    val_featured = add_aggregate_features(val_featured, train_split)

    train_featured = add_advanced_features(train_featured, train_split)
    val_featured = add_advanced_features(val_featured, train_split)

    train_featured = handle_missing_values(train_featured, train_split)
    val_featured = handle_missing_values(val_featured, train_split)

    # Финальные фичи
    exclude = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [c for c in train_featured.columns if c not in exclude and c not in ['timestamp']]

    X_train = train_featured[features]
    y_train = train_featured[config.TARGET]
    X_val = val_featured[features]
    y_val = val_featured[config.TARGET]

    print(f"Final features: {len(features)}")

    # === АНСАМБЛЬ ===
    val_preds = np.zeros(len(X_val))
    models = []
    weights = []

    # 1–3. LightGBM ×3
    for i, seed in enumerate([42, 123, 2025]):
        print(f"Training LGBM {i+1}/3 (seed={seed})...")
        model = lgb.LGBMRegressor(**config.LGB_PARAMS, n_estimators=10000, random_state=seed)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )
        pred = model.predict(X_val)
        val_preds += pred
        models.append(model)
        weights.append(1.0)
        print(f"   RMSE: {np.sqrt(mean_squared_error(y_val, pred)):.5f}")

    # 4. XGBoost
    print("Training XGBoost...")
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    xgb_model = xgb.train({
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'learning_rate': 0.02, 'max_depth': 10, 'subsample': 0.8,
        'colsample_bytree': 0.7, 'seed': 42, 'tree_method': 'hist'
    }, dtrain, num_boost_round=5000,
        evals=[(dval, 'val')], early_stopping_rounds=120, verbose_eval=False)
    pred = xgb_model.predict(dval)
    val_preds += pred
    models.append(xgb_model)
    weights.append(1.2)

    # 5. CatBoost
    print("Training CatBoost...")
    cat_features_idx = [i for i, c in enumerate(features) if c in config.CAT_FEATURES]
    cb_model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=10,
        loss_function='RMSE',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200,
        cat_features=cat_features_idx
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    pred = cb_model.predict(X_val)
    val_preds += 1.5 * pred
    models.append(cb_model)
    weights.append(1.5)

    # Финальный бленд
    final_val_pred = val_preds / sum(weights)
    rmse = np.sqrt(mean_squared_error(y_val, final_val_pred))
    mae = mean_absolute_error(y_val, final_val_pred)
    score = 1 - 0.5 * (rmse / 10 + mae / 10)
    print(f"\nFINAL VALIDATION → RMSE: {rmse:.5f} | MAE: {mae:.5f} | SCORE: {score:.5f} 🔥")

    # Сохраняем всё
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, config.MODEL_DIR / "ensemble_models.pkl")
    joblib.dump(features, config.MODEL_DIR / "features_list.pkl")
    joblib.dump(weights, config.MODEL_DIR / "ensemble_weights.pkl")
    print("Ensemble saved!")


if __name__ == "__main__":
    train()
