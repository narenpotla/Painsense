"""painsense.utils - seed_everything: make every run reproducible."""
import os
import random

import numpy as np
import torch

from painsense import config


def seed_everything(seed=None):
    """Seed python, numpy and torch (CPU + CUDA). Returns the seed used."""
    if seed is None:
        seed = config.SEED
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
