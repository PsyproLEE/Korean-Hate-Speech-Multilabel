import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dataset import TextDataset
from src.model import TextClassifier
from src.trainer import Trainer
from src.utils import load_config, get_device
from tools.torch_setup import setup_5080
import os
from datetime import datetime

def main():
    setup_5080()
    cfg = load_config("config/text_only_7cls.yaml")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(cfg["runtime"]["save_dir"], timestamp)
    os.makedirs(save_dir, exist_ok=True)

    cfg["runtime"]["save_best"] = os.path.join(save_dir, "best.pt")
    cfg["runtime"]["save_last"] = os.path.join(save_dir, "last.pt")

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["model_name"]
    )

    train_ds = TextDataset(
        cfg["data"]["train_csv"],
        tokenizer,
        cfg["data"]["text_col"],
        cfg["data"]["label_cols"],
        cfg["data"]["max_len"]
    )

    val_ds = TextDataset(
        cfg["data"]["valid_csv"],
        tokenizer,
        cfg["data"]["text_col"],
        cfg["data"]["label_cols"],
        cfg["data"]["max_len"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"]
    )

    model = TextClassifier(
        cfg["model"]["model_name"],
        len(cfg["data"]["label_cols"])
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = float(cfg["train"]["lr"])
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    trainer = Trainer(model, optimizer, criterion, device)
    best_f1 = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_f1 = trainer.eval_epoch(val_loader)
        print(
            f"[Epoch {epoch + 1}] "
            f"train_loss={train_loss:.4f}, "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                model.state_dict(),
                cfg["runtime"]["save_best"]
            )

    torch.save(
        model.state_dict(),
        cfg["runtime"]["save_last"]
    )


if __name__ == "__main__":
    main()
