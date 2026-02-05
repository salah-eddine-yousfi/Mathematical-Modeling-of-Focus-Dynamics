from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque
import cv2
import numpy as np


@dataclass
class CameraConfig:
    device_index: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    window_name: str = "Study Activity Recognition - press q to stop"
    show_window: bool = True
    window_width: int = 1000
    window_height: int = 700
    conf_threshold: float = 0.60
    streak: int = 3
    conf_alpha: float = 0.25
    show_hud: bool = True


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la webcam (device_index={cfg.device_index}).")

        if cfg.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        if cfg.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

        self._quit = False
        self._conf_smooth: float = 0.0
        self._hist: Deque[str] = deque(maxlen=max(int(cfg.streak), 1))
        self._stable_label: str = "unknown"
        self._last_label: str = "unknown"
        self._last_conf: float = 0.0

        self._display_name = {
            "focused_writing": "Writing Attention",
            "focused_reading": "Reading Attention",
            "not_phone": "Digital Distraction",
            "not_activity": "Cognitive Inactivity",
            "unknown": "Unknown",
        }

        self._title_color = {
            "focused_writing": (200, 120, 40),
            "focused_reading": (80, 180, 80),
            "not_phone": (60, 60, 220),
            "not_activity": (160, 160, 160),
            "unknown": (200, 200, 200),
        }

        if self.cfg.show_window:
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cfg.window_name, cfg.window_width, cfg.window_height)

    def read(self):
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("Erreur lecture frame webcam.")
        return frame_bgr

    def should_quit(self) -> bool:
        return self._quit

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            if self.cfg.show_window:
                cv2.destroyAllWindows()

    def set_hud(self, pred_class: str, conf: float):
        self._last_label = str(pred_class)
        self._last_conf = float(conf)

    @staticmethod
    def _put_text_outline(img, text, org, font_scale=0.9, color=(255, 255, 255), thickness=2):
        x, y = org
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_conf_bar(img, x, y, w, h, conf, th=0.60):
        conf = float(np.clip(conf, 0.0, 1.0))
        fill_w = int(w * conf)

        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 25, 25), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (235, 235, 235), 1)

        if conf >= th:
            col = (80, 220, 140)
        elif conf >= th * 0.85:
            col = (0, 190, 255)
        else:
            col = (80, 80, 255)

        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x + fill_w, y + h), col, -1)

    def _apply_hud(self, frame_bgr):
        conf = float(np.clip(self._last_conf, 0.0, 1.0))
        a = float(self.cfg.conf_alpha)
        self._conf_smooth = a * conf + (1.0 - a) * self._conf_smooth
        label = self._last_label if conf >= float(self.cfg.conf_threshold) else "unknown"

        self._hist.append(label)
        if len(self._hist) == self._hist.maxlen and len(set(self._hist)) == 1:
            self._stable_label = self._hist[0]

        title = self._display_name.get(self._stable_label, self._stable_label).upper()
        col = self._title_color.get(self._stable_label, (255, 255, 255))

        title_x, title_y = 30, 55
        self._put_text_outline(frame_bgr, title, (title_x, title_y),
                               font_scale=0.75, color=col, thickness=2)

        row_y = title_y + 28
        cv2.putText(frame_bgr, "Confidence", (title_x, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        bar_x = title_x + 110
        bar_y = row_y - 10
        self._draw_conf_bar(frame_bgr, bar_x, bar_y, 160, 10,
                            self._conf_smooth, th=float(self.cfg.conf_threshold))

        cv2.putText(frame_bgr, f"{int(self._conf_smooth * 100)}%",
                    (bar_x + 160 + 12, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 1, cv2.LINE_AA)

    def show(self, frame_bgr):
        if not self.cfg.show_window:
            return

        if self.cfg.show_hud:
            self._apply_hud(frame_bgr)

        cv2.imshow(self.cfg.window_name, frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._quit = True
