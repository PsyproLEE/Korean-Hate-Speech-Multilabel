import torch
from sklearn.metrics import f1_score

def multilabel_f1(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).int().cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

