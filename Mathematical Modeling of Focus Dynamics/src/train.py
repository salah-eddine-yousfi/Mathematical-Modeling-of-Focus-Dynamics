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

def run_epoch(model, loader, device, crit, opt=None):
    if opt is None:
        model.eval()
    else:
        model.train()

    loss_sum, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if opt is not None:
                opt.zero_grad()

            logits = model(x)
            loss = crit(logits, y)

            if opt is not None:
                loss.backward()
                opt.step()

            bs = y.size(0)
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            total += bs

    return loss_sum / max(total, 1), correct / max(total, 1)

def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yload(root / "config.yaml")
    device = device_of(cfg.get("device", "auto"))

    tr = get_loader("train")
    va = get_loader("val")

    model = build_model(cfg, device)
    crit = nn.CrossEntropyLoss()

    weight_decay = float(cfg.get("weight_decay", 1e-4))
    freeze_epochs = int(cfg.get("freeze_epochs", 2))

    if freeze_epochs > 0:
        freeze_backbone(model)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["learning_rate"]),
        weight_decay=weight_decay
    )

    best_val = float("inf")
    (root / "results").mkdir(exist_ok=True)

    print(f"Device: {device}")
    print(f"freeze_epochs={freeze_epochs} | weight_decay={weight_decay}")
    print("epoch | train_loss train_acc | val_loss val_acc")
    print("-" * 55)

    for e in range(int(cfg["epochs"])):
        if freeze_epochs > 0 and e == freeze_epochs:
            unfreeze_all(model)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg["learning_rate"]),
                weight_decay=weight_decay
            )
            print(">> Unfroze backbone (now training all layers)")

        train_loss, train_acc = run_epoch(model, tr, device, crit, opt=opt)
        val_loss, val_acc     = run_epoch(model, va, device, crit, opt=None)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), root / "results" / "best.pt")

        print(f"{e+1:>5} | {train_loss:>10.4f} train_acc: {train_acc:.3f} | {val_loss:>8.4f} val_acc: {val_acc:.3f}")

if __name__ == "__main__":
    main()
