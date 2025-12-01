import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import catboost as cb
import optuna

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def prepare_data():
    path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Нет обработанных данных: {path}")

    print("Грузим данные...")
    df = pd.read_parquet(path)

    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # темпоральный сплит
    split_date = get_split_date_from_ratio(train_df, config.TEMPORAL_SPLIT_RATIO)
    train_mask, val_mask = temporal_split_by_date(train_df, split_date)

    train = train_df[train_mask].copy()
    val = train_df[val_mask].copy()

    print(f"Трейн: {len(train):,} | Вал: {len(val):,}")

    # агрегаты без лика
    train = add_aggregate_features(train, train)
    val = add_aggregate_features(val, train)

    for d in [train, val]:
        d.drop(columns=["time_decay"], errors="ignore", inplace=True)

    # пропуски
    train = handle_missing_values(train, train)
    val = handle_missing_values(val, train)

    # список фичей
    exclude = {constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP}
    features = [c for c in train.columns if c not in exclude and train[c].dtype.name != "object"]
    cat_features = [c for c in features if train[c].dtype.name == "category"]

    X_train, y_train = train[features], train[config.TARGET]
    X_val, y_val = val[features], val[config.TARGET]

    print(f"Фичей: {len(features)} (категорий: {len(cat_features)})")

    return X_train, y_train, X_val, y_val, features, cat_features


def objective(trial, X_tr, y_tr, X_val, y_val, cat_feats):
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2", 1, 20),
        "bagging_temperature": trial.suggest_float("bag", 0.5, 2),
        "random_strength": trial.suggest_float("rs", 0.5, 2),
        "border_count": trial.suggest_int("bc", 32, 255),
        "min_data_in_leaf": trial.suggest_int("min_leaf", 1, 10),
        "loss_function": "RMSEWithUncertainty",
        "eval_metric": "RMSE",
        "od_type": "Iter",
        "od_wait": config.EARLY_STOPPING_ROUNDS,
        "random_seed": config.RANDOM_STATE,
        "verbose": 0,
        "thread_count": -1,
        # "task_type": "GPU", "devices": "0"
    }

    pool_tr = cb.Pool(X_tr, y_tr, cat_features=cat_feats)
    pool_val = cb.Pool(X_val, y_val, cat_features=cat_feats)

    model = cb.CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_val, use_best_model=True)

    pred = model.predict(X_val)
    if pred.ndim == 2:
        pred = pred[:, 0]

    return np.sqrt(mean_squared_error(y_val, pred))


def train(do_tune=False):
    X_train, y_train, X_val, y_val, features, cat_features = prepare_data()

    if do_tune:
        print("\nЗапускаем тюнинг optuna...")
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
        study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, cat_features),
                       n_trials=50, show_progress_bar=True)

        print(f"Лучший RMSE: {study.best_value:.5f}")
        best = study.best_params
        params = {
            "iterations": 3000,
            "learning_rate": best["lr"],
            "depth": best["depth"],
            "l2_leaf_reg": best["l2"],
            "bagging_temperature": best["bag"],
            "random_strength": best["rs"],
            "border_count": best["bc"],
            "min_data_in_leaf": best["min_leaf"],
            "loss_function": "RMSEWithUncertainty",
            "eval_metric": "RMSE",
            "od_type": "Iter",
            "od_wait": config.EARLY_STOPPING_ROUNDS,
            "random_seed": config.RANDOM_STATE,
            "verbose": 200,
            "thread_count": -1,
            "task_type": "GPU", "devices": "0"
        }
    else:
        # просто хорошие параметры, которые уже работали
        params = {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "border_count": 254,
            "min_data_in_leaf": 5,
            "loss_function": "RMSEWithUncertainty",
            "eval_metric": "RMSE",
            "od_type": "Iter",
            "od_wait": config.EARLY_STOPPING_ROUNDS,
            "random_seed": config.RANDOM_STATE,
            "verbose": 200,
            "thread_count": -1,
        }

    print("\nСтартуем обучение...")
    pool_train = cb.Pool(X_train, y_train, cat_features=cat_features)
    pool_val = cb.Pool(X_val, y_val, cat_features=cat_features)

    model = cb.CatBoostRegressor(**params)
    model.fit(pool_train, eval_set=pool_val, use_best_model=True)

    pred_val = model.predict(X_val)
    if pred_val.ndim == 2:
        pred_val = pred_val[:, 0]

    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    mae = mean_absolute_error(y_val, pred_val)
    print(f"\nRMSE на валидации: {rmse:.5f}")
    print(f"MAE на валидации:  {mae:.5f}")

    # сохраняем
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.MODEL_DIR / "catboost_model.cbm"))
    joblib.dump(features, config.MODEL_DIR / "feature_columns.pkl")

    print(f"Модель сохранена → {config.MODEL_DIR / 'catboost_model.cbm'}")
    print("Всё готово!")


if __name__ == "__main__":
    train(do_tune=True)
