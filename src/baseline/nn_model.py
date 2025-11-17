import torch
from pytorch_tabnet.tab_model import TabNetRegressor

clf = TabNetRegressor(
    n_d=64, n_a=64,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 50, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    verbose=10
)

clf.fit(
    X_train.values, y_train.values.reshape(-1, 1),
    eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
    max_epochs=200,
    patience=50,
    batch_size=1024,
    virtual_batch_size=128
)
