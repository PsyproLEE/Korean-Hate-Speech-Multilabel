import torch
from tqdm import tqdm
from .metrics import multilabel_f1


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(loader):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def eval_epoch(self, loader):
        self.model.eval()
        total_loss, total_f1 = 0, 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, mask)
                loss = self.criterion(logits, labels)
                f1 = multilabel_f1(logits, labels)

                total_loss += loss.item()
                total_f1 += f1

        return total_loss / len(loader), total_f1 / len(loader)
