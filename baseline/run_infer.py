import torch
from transformers import AutoTokenizer
from src.model import TextClassifier
from src.inference import predict
from src.utils import load_config, get_device

def main():
    # load config
    cfg = load_config("config/text_only_7cls.yaml")
    device = get_device()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # model
    model = TextClassifier(
        model_name=cfg["model_name"],
        num_labels=len(cfg["label_cols"])
    )
    model.load_state_dict(torch.load(cfg["model_ckpt"], map_location=device))
    model.to(device)

    # example inputs
    texts = [
        "너 진짜 왜 그러냐",
        "이런 말은 하면 안 되지"
    ]

    preds = predict(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        threshold=cfg.get("threshold", 0.5)
    )

    # output
    for text, pred in zip(texts, preds):
        active_labels = [
            label for label, v in zip(cfg["label_cols"], pred) if v == 1
        ]
        print(f"Input: {text}")
        print(f"Predicted labels: {active_labels}")
        print("-" * 40)

if __name__ == "__main__":
    main()
