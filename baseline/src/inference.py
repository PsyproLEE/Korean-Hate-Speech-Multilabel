import torch

def predict(model, tokenizer, texts, device, threshold=0.5):
    model.eval()
    results = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)

            logits = model(**enc)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            results.append((probs > threshold).astype(int))

    return results
