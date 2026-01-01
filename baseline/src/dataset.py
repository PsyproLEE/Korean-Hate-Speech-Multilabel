import torch
from torch.utils.data import Dataset
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, text_col, label_cols, max_len):
        self.df = pd.read_csv(csv_path)
        self.text_col = text_col
        self.label_cols = label_cols
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, self.text_col]
        labels = self.df.loc[idx, self.label_cols].values.astype(float)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels)
        }

        p
