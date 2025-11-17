# models/ft_transformer.py — SOTA Neural Net

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from typing import Dict, List

from . import config, constants  # Подтяни из твоего проекта

class FTTransformer(nn.Module):
    def __init__(self, num_features: int, cat_cardinalities: List[int], d_token: int = 256, n_blocks: int = 4, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_embed = nn.Linear(num_features, d_token) if num_features > 0 else None
        self.cat_embeds = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cardinalities]) if cat_cardinalities else None

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

    def forward(self, x_num: torch.Tensor = None, x_cat: torch.Tensor = None) -> torch.Tensor:
        embeds = []
        if self.num_embed and x_num is not None:
            embeds.append(self.num_embed(x_num))
        if self.cat_embeds and x_cat is not None:
            for i, emb in enumerate(self.cat_embeds):
                embeds.append(emb(x_cat[:, i]))
        x = torch.stack(embeds, dim=1)  # [B, N_feats, D]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global Avg Pool
        return self.head(x).squeeze(1)

class RatingDataset(Dataset):
    def __init__(self, X_num: np.ndarray = None, X_cat: np.ndarray = None, y: np.ndarray = None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32) if X_num is not None else None
        self.X_cat = torch.tensor(X_cat, dtype=torch.long) if X_cat is not None else None
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X_num if self.X_num is not None else self.X_cat)

    def __getitem__(self, idx: int) -> tuple:
        item = []
        if self.X_num is not None: item.append(self.X_num[idx])
        if self.X_cat is not None: item.append(self.X_cat[idx])
        if self.y is not None: item.append(self.y[idx])
        return tuple(item)

def prepare_data_for_nn(df: pd.DataFrame, cat_features: List[str], num_features: List[str], fit: bool = False, encoders: Dict = None, scaler: StandardScaler = None) -> tuple:
    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[num_features].fillna(0).values)
        encoders = {}
    else:
        X_num = scaler.transform(df[num_features].fillna(0).values)

    X_cat = np.zeros((len(df), len(cat_features)), dtype=np.int64)
    for i, col in enumerate(cat_features):
        if fit:
            le = LabelEncoder()
            le.fit(df[col].astype(str).fillna('missing'))
            encoders[col] = le
        else:
            le = encoders[col]
        X_cat[:, i] = le.transform(df[col].astype(str).fillna('missing'))

    cardinalities = [len(le.classes_) for le in encoders.values()] if fit else None
    return X_num, X_cat, cardinalities if fit else None, encoders if fit else None, scaler if fit else None

def train_ft_transformer(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, cat_features: List[str], num_features: List[str]) -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"FT-Transformer on {device}...")

    X_train_num, X_train_cat, cardinalities, encoders, scaler = prepare_data_for_nn(X_train, cat_features, num_features, fit=True)
    X_val_num, X_val_cat, _, _, _ = prepare_data_for_nn(X_val, cat_features, num_features, fit=False, encoders=encoders, scaler=scaler)

    train_ds = RatingDataset(X_train_num, X_train_cat, y_train.values)
    val_ds = RatingDataset(X_val_num, X_val_cat, y_val.values)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4)

    model = FTTransformer(len(num_features), cardinalities).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=20, steps_per_epoch=len(train_loader))

    best_rmse = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(20):
        model.train()
        for batch in train_loader:
            x_num, x_cat, y = [b.to(device) for b in batch]
            pred = model(x_num, x_cat)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                x_num, x_cat, _ = [b.to(device) for b in batch]
                val_preds.append(model(x_num, x_cat).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        print(f"Epoch {epoch+1:02d} | Val RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            patience_counter = 0
            torch.save({'model': model.state_dict(), 'encoders': encoders, 'scaler': scaler}, config.MODEL_DIR / 'ft_transformer.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    return model  # Для predict используем сохранённый
