import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from NewTFTModel import TemporalFusionTransformer 
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss


data = pd.read_csv('datasets/merge/result.csv', index_col = False)
data['time_idx'] = data.index
data['group'] = 0

max_prediction_length = 7
max_encoder_length = 24
training_cutoff = data['time_idx'].max() - max_prediction_length

# Training set and validation set
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    group_ids = ['group'],
    target='market_price',
    time_idx='time_idx',
    add_relative_time_idx=True,
    add_target_scales=True,
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=['ave_block_size', 'difficulty', 'hash_rate', 'market_price',
                                'miners_rev', 'transaction', 'ex_trade_vol']
)
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# Dataloaders
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

#Train the model
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=30,
    gpus=1,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

trainer.fit(
    tft, 
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)