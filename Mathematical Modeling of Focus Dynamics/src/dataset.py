from pathlib import Path
from PIL import Image
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def yload(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))

class ImgDataset(Dataset):
    def __init__(self, root, classes, tfm):
        self.x, self.y = [], []
        for i, c in enumerate(classes):
            cls_dir = root / c
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder in split: {cls_dir}")
            for p in cls_dir.iterdir():
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.x.append(p)
                    self.y.append(i)
        self.tfm = tfm

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = Image.open(self.x[i]).convert("RGB")
        return self.tfm(img), self.y[i]

def get_loader(split: str):
    root = Path(__file__).resolve().parents[1]
    cfg = yload(root / "config.yaml")

    img_size = int(cfg["img_size"])

    if split == "train":
        tfm = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    ds = ImgDataset(
        root / "data" / "splits" / split,
        cfg["classes"],
        tfm
    )

    return DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=(split == "train"),
        num_workers=int(cfg["num_workers"])
    )
