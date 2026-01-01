import yaml
import torch

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
