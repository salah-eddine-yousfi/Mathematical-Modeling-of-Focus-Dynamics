from pathlib import Path
import yaml, torch
import torch.nn as nn
from src.dataset import get_loader
from src.model import build_model

def yload(p): 
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def device_of(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yload(root / "config.yaml")
    device = device_of(cfg.get("device","auto"))

    te = get_loader("test")
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(root / "results" / "best.pt", map_location=device))
    model.eval()

    crit = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x,y in te:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += crit(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    print("test_loss:", loss_sum / max(total,1))
    print("test_acc :", correct / max(total,1))

if __name__ == "__main__":
    main()
