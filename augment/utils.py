import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from augment.core.dataset import HateDataset, load_and_split


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataloaders(cfg):
    csv_path = cfg.get("data", "csv_path")
    max_len = cfg.get("data", "max_len", default=128)

    train_df, val_df, test_df = load_and_split(
        csv_path,
        train_ratio=cfg.get("data", "train_ratio", default=0.8),
        val_ratio=cfg.get("data", "val_ratio", default=0.1),
        seed=cfg.seed,
        use_augment=cfg.get("augment", "use_augment", default=False),
        max_aug_per_sample=cfg.get("augment", "max_aug_per_sample", default=2),
        apply_to_hate_only=cfg.get("augment", "apply_to_hate_only", default=True),
    )

    train_ds = HateDataset(train_df, cfg.get("model", "plm_name"), max_len)
    val_ds   = HateDataset(val_df,   cfg.get("model", "plm_name"), max_len)
    test_ds  = HateDataset(test_df,  cfg.get("model", "plm_name"), max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.get("train","batch_size"), shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.get("train","batch_size"), shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.get("train","batch_size"), shuffle=False)

    return train_loader, val_loader, test_loader, train_df