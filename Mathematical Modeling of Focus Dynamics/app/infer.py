from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from src.model import build_model
from app.preprocess import PreprocessConfig, Preprocessor


@dataclass
class InferenceOutput:
    pred_class: str
    pred_proba: float
    pred_index: int


class InferenceEngine:
    def __init__(
        self,
        cfg: dict,
        classes: List[str],
        best_pt_path: Path,
        device: torch.device,
        preprocess_cfg: PreprocessConfig,
    ):
        self.cfg = cfg
        self.classes = classes
        self.device = device

        if not best_pt_path.exists():
            raise FileNotFoundError(f"best.pt introuvable: {best_pt_path}")

        self.model = build_model(cfg, device)
        state = torch.load(best_pt_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.preprocessor = Preprocessor(preprocess_cfg, device=device)

    @torch.no_grad()
    def predict(self, frame_bgr) -> InferenceOutput:
        x = self.preprocessor(frame_bgr)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probs).item())
        pred_proba = float(probs[pred_idx].item())
        pred_class = self.classes[pred_idx]

        return InferenceOutput(pred_class=pred_class, pred_proba=pred_proba, pred_index=pred_idx)
