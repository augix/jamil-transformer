#!/usr/bin/env python
# %%
from train import train_model
from config import get_config

# additional config
cfg = get_config()
cfg['batch_size'] = 6
cfg['preload'] = None
cfg['num_epochs'] = 20

train_model(cfg)

# %%



