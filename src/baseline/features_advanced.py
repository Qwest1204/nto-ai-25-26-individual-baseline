import numpy as np
import pandas as pd

from . import constants, config


def add_advanced_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    print("=== Adding ADVANCED features (this is where the magic happens) ===")
    tr = train_df.copy()
    full = df.copy()

    # Convert timestamp to seconds
    tr['ts'] = tr[constants.COL_TIMESTAMP].astype('int64') // 10**9
    full['ts'] = full[constants.COL_TIMESTAMP].astype('int64') // 10**9

    global_mean = tr[config.TARGET].mean()

    # ========================================
    # 1. User temporal & dynamics features
    # ========================================
    tr_sorted = tr.sort_values(['user_id', 'timestamp'])

    # Previous ratings
    tr_sorted['user_prev_rating'] = tr_sorted.groupby('user_id')[config.TARGET].shift(1)
    tr_sorted['user_prev2_rating'] = tr_sorted.groupby('user_id')[config.TARGET].shift(2)
    tr_sorted['user_rating_diff'] = tr_sorted[config.TARGET] - tr_sorted['user_prev_rating']
    tr_sorted['user_rating_diff2'] = tr_sorted['user_prev_rating'] - tr_sorted['user_prev2_rating']

    user_stats = tr_sorted.groupby('user_id').agg(
        user_last_rating=('rating', 'last'),
        user_second_last_rating=('rating', lambda x: x.iloc[-2] if len(x) >= 2 else np.nan),
        user_third_last_rating=('rating', lambda x: x.iloc[-3] if len(x) >= 3 else np.nan),
        user_rating_trend=('user_rating_diff', 'mean'),
        user_rating_trend2=('user_rating_diff2', 'mean'),
        user_rating_volatility=('user_rating_diff', 'std'),
        user_rating_volatility_abs=('user_rating_diff', lambda x: x.abs().mean()),
        user_rating_skew=('rating', 'skew'),
        user_first_ts=('ts', 'min'),
        user_last_ts=('ts', 'max'),
        user_active_days=('ts', lambda x: (x.max() - x.min()) / 86400 + 1),
        user_ratings_count=('rating', 'count'),
    ).reset_index()

    user_stats['user_rating_frequency'] = user_stats['user_ratings_count'] / user_stats['user_active_days']
    user_stats['user_recency_score'] = np.exp(-(tr['ts'].max() - user_stats['user_last_ts']) / 86400 / 30)  # last month bonus

    # ========================================
    # 2. Time gaps (очень важно для теста!)
    # ========================================
    last_user_activity = tr_sorted.groupby('user_id')['ts'].max().rename('user_last_activity_ts')
    approx_test_ts = full[full[constants.COL_SOURCE] == 'test']['ts'].min()
    user_gap = pd.DataFrame({'user_id': last_user_activity.index})
    user_gap['days_since_last_rating'] = (approx_test_ts - last_user_activity.values) / 86400
    user_gap['days_since_last_rating_log'] = np.log1p(user_gap['days_since_last_rating'])

    # ========================================
    # 3. Smoothed target encodings (гораздо лучше простого mean)
    # ========================================
    def smoothed_te(col, alpha=100):
        agg = tr.groupby(col)['rating'].agg(['mean', 'count'])
        smoothed = (agg['mean'] * agg['count'] + global_mean * alpha) / (agg['count'] + alpha)
        return smoothed.to_dict()

    te_map = {
        'author_id': smoothed_te('author_id', 150),
        'publisher': smoothed_te('publisher', 80),
        'language': smoothed_te('language', 50),
        'publication_year': smoothed_te('publication_year', 30),
    }

    for col, mapping in te_map.items():
        full[f'{col}_smoothed_rating'] = full[col].map(mapping).fillna(global_mean)

    # ========================================
    # 4. User preferences per category
    # ========================================
    user_author_pref = tr.groupby(['user_id', 'author_id'])['rating'].mean().groupby('user_id').mean()
    user_lang_pref = tr.groupby(['user_id', 'language'])['rating'].mean().groupby('user_id').mean()

    full = full.merge(user_author_pref.rename('user_author_bias'), on='user_id', how='left')
    full = full.merge(user_lang_pref.rename('user_lang_bias'), on='user_id', how='left')

    # Deviation from user's average
    user_mean = tr.groupby('user_id')['rating'].mean()
    full = full.merge(user_mean.rename('user_global_mean'), on='user_id', how='left')
    full['user_vs_global_author'] = full['author_id_smoothed_rating'] - full['user_global_mean']
    full['user_vs_global_lang'] = full['language_smoothed_rating'] - full['user_global_mean']

    # ========================================
    # 5. Merge everything
    # ========================================
    merge_dfs = [user_stats, user_gap]
    for mdf in merge_dfs:
        full = full.merge(mdf, on='user_id', how='left')

    # Fill NaNs for new users (though problem says no cold users)
    fill_cols = [
        'user_last_rating', 'user_second_last_rating', 'user_third_last_rating',
        'user_rating_trend', 'user_rating_volatility', 'user_rating_skew',
        'user_rating_frequency', 'user_recency_score', 'days_since_last_rating',
        'user_author_bias', 'user_lang_bias', 'user_global_mean'
    ]
    for col in fill_cols:
        if col in full.columns:
            full[col] = full[col].fillna(full[col].median() if full[col].dtype != 'object' else global_mean)

    print(f"Advanced features added: {len(full.columns) - len(df.columns)} new columns")
    return full
