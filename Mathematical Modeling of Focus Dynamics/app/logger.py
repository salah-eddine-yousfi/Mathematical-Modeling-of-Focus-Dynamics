from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd


@dataclass
class SessionLogger:
    classes: List[str]
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        frame_idx: int,
        time_sec: float,
        class_raw: str,
        proba: float,
        state_raw: int,  # 1 / 0 / -1 (unknown)
        llr: float,
        L: float,
        p: float,
        state_filtered: int,
        focused_time_sec: float,
        non_focused_time_sec: float,
        is_unknown: int,  # 1/0 (plus simple pour Excel)
    ):
        self.rows.append({
            "frame_idx": frame_idx,
            "time_sec": time_sec,
            "class_raw": class_raw,
            "proba": proba,
            "state_raw": state_raw,
            "llr": llr,
            "L": L,
            "p": p,
            "state_filtered": state_filtered,
            "focused_time_sec": focused_time_sec,
            "non_focused_time_sec": non_focused_time_sec,
            "is_unknown": is_unknown,
        })

    def to_dataframe(self) -> pd.DataFrame:
        cols = [
            "frame_idx", "time_sec", "class_raw", "proba",
            "state_raw", "llr", "L", "p", "state_filtered",
            "focused_time_sec", "non_focused_time_sec",
            "is_unknown",
        ]
        if not self.rows:
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(self.rows)

        df["frame_idx"] = df["frame_idx"].astype(int)
        df["time_sec"] = df["time_sec"].astype(float)
        df["proba"] = df["proba"].astype(float)

        df["state_raw"] = df["state_raw"].astype(int)
        df["state_filtered"] = df["state_filtered"].astype(int)
        df["is_unknown"] = df["is_unknown"].astype(int)

        df["focused_time_sec"] = df["focused_time_sec"].astype(float)
        df["non_focused_time_sec"] = df["non_focused_time_sec"].astype(float)

        df["llr"] = df["llr"].astype(float)
        df["L"] = df["L"].astype(float)
        df["p"] = df["p"].astype(float)

        return df
