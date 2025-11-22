# src/baseline/train.py — полностью обновлённый под новый features.py

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor, Pool
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import DataLoader

from . import config, constants
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date
from .nn_model import train_ft_transformer, FTTransformer, prepare_data_for_nn, RatingDataset


def train() -> None:
    # Загрузка обработанных данных
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(f"Нет обработанных данных! Сначала запусти: python -m src.baseline.prepare_data")

    print("Загрузка обработанных данных...")
    df = pd.read_parquet(processed_path)
    print(f"Загружено {len(df):,} строк, {len(df.columns)} колонок")

    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Temporal split
    split_date = get_split_date_from_ratio(train_df, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Дата сплита: {split_date}")
    train_mask = train_df[constants.COL_TIMESTAMP] <= split_date
    val_mask = train_df[constants.COL_TIMESTAMP] > split_date

    train_split = train_df[train_mask].copy()
    val_split = train_df[val_mask].copy()

    print(f"Train: {len(train_split):,} | Val: {len(val_split):,}")

    # === ФИЧИ УЖЕ ЕСТЬ В ФАЙЛЕ! Ничего добавлять не нужно ===
    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_TIMESTAMP, "rating_predict"]
    features = [c for c in train_split.columns if c not in exclude_cols]

    X_train = train_split[features]
    y_train = train_split[config.TARGET].clip(0, 10)
    X_val = val_split[features]
    y_val = val_split[config.TARGET].clip(0, 10)

    cat_features = [c for c in config.CAT_FEATURES if c in features]

    # Приводим категории к строкам (CatBoost требует)
    for col in cat_features:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    print(f"Обучение на {len(features)} признаках ({len(cat_features)} категориальных)")

    # === CatBoost ===
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

    model.fit(Pool(X_train, y_train, cat_features=cat_features),
              eval_set=Pool(X_val, y_val, cat_features=cat_features))

    # === LightGBM ===
    print("Обучение LightGBM...")
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=cat_features)

    lgb_model = lgb.train(
        config.LGB_PARAMS,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(100)]
    )

    # === FT-Transformer ===
    print("Обучение FT-Transformer...")
    num_features = [c for c in features if c not in cat_features]
    train_ft_transformer(train_split[features], y_train, val_split[features], y_val, cat_features, num_features)

    # === Ансамбль на валидации ===
    cat_pred = model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val)

    # FT-Transformer предикт
    ft_state = torch.load(config.MODEL_DIR / "ft_transformer.pt", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_model = FTTransformer(len(num_features), [len(ft_state['encoders'][c].classes_) for c in cat_features]).to(device)
    ft_model.load_state_dict(ft_state['model'])
    ft_model.eval()

    X_val_num, X_val_cat, _, _, _ = prepare_data_for_nn(val_split, cat_features, num_features, fit=False,
                                                        encoders=ft_state['encoders'], scaler=ft_state['scaler'])
    val_ds = RatingDataset(X_val_num, X_val_cat)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

    ft_preds = []
    with torch.no_grad():
        for x_num, x_cat in val_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            ft_preds.append(ft_model(x_num, x_cat).cpu().numpy())
    ft_pred = np.concatenate(ft_preds)

    # Финальный ансамбль
    final_pred = 0.5 * cat_pred + 0.3 * ft_pred + 0.2 * lgb_pred
    rmse = np.sqrt(mean_squared_error(y_val, final_pred))
    print(f"\nENSEMBLE VAL RMSE: {rmse:.5f}")

    # Сохранение
    model.save_model(config.MODEL_DIR / "catboost_model.cbm")
    lgb_model.save_model(config.MODEL_DIR / "lightgbm_model.txt")
    print("Модели сохранены!")

if __name__ == "__main__":
    train()
