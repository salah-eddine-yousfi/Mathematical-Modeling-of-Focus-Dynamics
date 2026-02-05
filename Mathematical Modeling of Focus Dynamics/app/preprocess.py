from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2


@dataclass
class PreprocessConfig:
    img_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class Preprocessor:
  
    def __init__(self, cfg: PreprocessConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

       
        self.tfm = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ])

    def __call__(self, frame_bgr: np.ndarray) -> torch.Tensor:
       
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame_rgb)

        x = self.tfm(img)

        x = x.unsqueeze(0).to(self.device)

        return x
