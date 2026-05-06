import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, MultiLoss
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer


def train_tft_daily_multi48(
    X: np.ndarray,                  # (T, 248) — training features
    Y: np.ndarray,                  # (T, 48)  — training targets
    enc_len: int = 60,
    val_frac: float = 0.2,
    batch_size: int = 128,
    max_epochs: int = 50,
    hidden_size: int = 64,          
    attention_head_size: int = 4,
    lstm_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    patience: int = 6,
    seed: int = 7,
    known_feature_indices: list = None,
    unknown_feature_indices: list = None,
):
    
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    T, F = X.shape
    assert Y.shape == (T, 48), f"Expected Y shape (T, 48), got {Y.shape}"

    pl.seed_everything(seed, workers=True)

    
    feature_cols = [f"f{j}" for j in range(F)]
    target_cols = [f"y{k}" for k in range(48)]

    
    if known_feature_indices is None and unknown_feature_indices is None:
        
        known_cols = feature_cols
        unknown_cols = []
    else:
        known_idx = known_feature_indices or []
        unknown_idx = unknown_feature_indices or []
        # Validate no overlap
        overlap = set(known_idx) & set(unknown_idx)
        assert len(overlap) == 0, f"Indices in both known and unknown: {overlap}"
        known_cols = [f"f{j}" for j in known_idx]
        unknown_cols = [f"f{j}" for j in unknown_idx]

    # ── Build DataFrame ──
    df = pd.DataFrame(X, columns=feature_cols)
    df["time_idx"] = np.arange(T, dtype=np.int64)
    df["series"] = 0
    for k in range(48):
        df[f"y{k}"] = Y[:, k]

    # ── Train/val split by time ──
    val_len = int(max(10, np.floor(val_frac * T)))
    cutoff = T - val_len - 1

    target_norm = MultiNormalizer([TorchNormalizer(method="identity")] * 48)

    training = TimeSeriesDataSet(
        df[df.time_idx <= cutoff],
        time_idx="time_idx",
        target=target_cols,
        group_ids=["series"],
        max_encoder_length=enc_len,
        max_prediction_length=1,
        time_varying_known_reals=["time_idx"] + known_cols,
        time_varying_unknown_reals=unknown_cols,
        target_normalizer=target_norm,
        add_relative_time_idx=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=cutoff + 1, stop_randomization=True
    )

    train_dl = training.to_dataloader(train=True,  batch_size=batch_size,     num_workers=0)
    val_dl   = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    # ── Model ──
    loss = MultiLoss([MAE()] * 48)
    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        loss=loss,
        output_size=[1] * 48,
        log_interval=-1,
        reduce_on_plateau_patience=5,
    )

    # ── Train ──
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, mode="min")],
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    return model, training


def predict_tft_next_day(
    model: TemporalFusionTransformer,
    training_dataset: TimeSeriesDataSet,
    X_history: np.ndarray,          # (enc_len, 248) — last enc_len days of features
    X_new: np.ndarray,              # (248,) — features for the day to predict
    enc_len: int = 60,
):
    
    X_history = np.asarray(X_history, dtype=np.float32)
    X_new = np.asarray(X_new, dtype=np.float32).reshape(1, -1)
    F = X_history.shape[1]

    assert X_history.shape == (enc_len, F), (
        f"X_history must be ({enc_len}, {F}), got {X_history.shape}"
    )
    assert X_new.shape == (1, F), f"X_new must be ({F},), got {X_new.shape}"

    
    feature_cols = [f"f{j}" for j in range(F)]
    target_cols = [f"y{k}" for k in range(48)]

    X_all = np.vstack([X_history, X_new])       # (enc_len + 1, F)
    n_rows = X_all.shape[0]

    df_pred = pd.DataFrame(X_all, columns=feature_cols)
    df_pred["time_idx"] = np.arange(n_rows, dtype=np.int64)
    df_pred["series"] = 0

    # Targets must exist as columns — fill with dummy values
    for k in range(48):
        df_pred[f"y{k}"] = 0.0

    
    pred_ds = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df_pred,
        predict=True,
        stop_randomization=True,
    )
    pred_dl = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    
    pred = model.predict(pred_dl)

    if isinstance(pred, list):
        # Multi-target: list of 48 tensors, each (n_samples, 1)
        pred = torch.stack([p.detach().cpu() for p in pred], dim=-1)
        yhat = pred.numpy()
    else:
        yhat = pred.detach().cpu().numpy()

    
    # Shape is (n_samples, 1, 48) — with predict=True and 1 group, n_samples=1
    y_hat = yhat[0, 0, :]     # (48,)

    return y_hat
