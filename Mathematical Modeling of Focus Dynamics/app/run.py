from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import time
import argparse
import yaml
import torch
from app.camera import Camera, CameraConfig
from app.preprocess import PreprocessConfig
from app.infer import InferenceEngine
from app.mapping import map_class_to_binary_or_none
from app.evidence_model import EvidenceConfig, EvidenceAccumulator
from app.logger import SessionLogger
from app.plots import generate_all_plots


def yload(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def device_of(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def make_session_dir(root_results: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = root_results / "sessions" / ts
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


@dataclass
class AppParams:
    frame_interval_sec: float = 3.0
    session_duration_sec: int | None = None
    camera_index: int = 0

    H: float = 0.03
    lambda_on: float = 0.40
    lambda_off: float = 1.10
    lambda_unknown: float = 0.0
    threshold: float = 0.6

    conf_threshold: float = 0.60
    streak: int = 3
    conf_alpha: float = 0.25


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--duration", type=int, default=None)
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--camera", type=int, default=None)

    ap.add_argument("--H", type=float, default=None)
    ap.add_argument("--lambda_on", type=float, default=None)
    ap.add_argument("--lambda_off", type=float, default=None)
    ap.add_argument("--lambda_unknown", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None)

    ap.add_argument("--conf_threshold", type=float, default=None)
    ap.add_argument("--streak", type=int, default=None)
    ap.add_argument("--conf_alpha", type=float, default=None)

    return ap.parse_args()


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yload(root / "config.yaml")
    device = device_of(cfg.get("device", "auto"))

    args = parse_args()
    P = AppParams()

    if args.duration is not None:
        P.session_duration_sec = args.duration
    if args.interval is not None:
        P.frame_interval_sec = args.interval
    if args.camera is not None:
        P.camera_index = args.camera

    if args.H is not None:
        P.H = args.H
    if args.lambda_on is not None:
        P.lambda_on = args.lambda_on
    if args.lambda_off is not None:
        P.lambda_off = args.lambda_off
    if args.lambda_unknown is not None:
        P.lambda_unknown = args.lambda_unknown
    if args.threshold is not None:
        P.threshold = args.threshold

    if args.conf_threshold is not None:
        P.conf_threshold = args.conf_threshold
    if args.streak is not None:
        P.streak = args.streak
    if args.conf_alpha is not None:
        P.conf_alpha = args.conf_alpha

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    session_dir = make_session_dir(results_dir)

    pp_cfg = PreprocessConfig(
        img_size=int(cfg["img_size"]),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    classes: list[str] = list(cfg["classes"])

    best_pt = root / "results" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    infer_engine = InferenceEngine(
        cfg=cfg,
        classes=classes,
        best_pt_path=best_pt,
        device=device,
        preprocess_cfg=pp_cfg,
    )

    ev_cfg = EvidenceConfig(
        H=P.H,
        lambda_on=P.lambda_on,
        lambda_off=P.lambda_off,
        lambda_unknown=P.lambda_unknown,
        threshold=P.threshold,
        L0=0.0,
    )
    accumulator = EvidenceAccumulator(ev_cfg)

    logger = SessionLogger(classes=classes)

    cam_cfg = CameraConfig(
        device_index=P.camera_index,
        show_window=True,
        window_width=1000,
        window_height=700,
        show_hud=True,
        conf_threshold=P.conf_threshold,
        streak=P.streak,
        conf_alpha=P.conf_alpha,
    )
    cam = Camera(cam_cfg)

    print("\n[APP] Session dir :", session_dir)
    print("[APP] Device      :", device)
    print("[APP] Inference   :", P.frame_interval_sec, "sec")
    print("[APP] H           :", P.H)
    print("[APP] p-threshold :", P.threshold)
    print("[APP] conf-thresh :", P.conf_threshold, "(unknown if below)")
    print("[APP] Quit        : press 'q'\n")

    start_t = time.time()
    next_infer_t = start_t
    last_infer_time: float | None = None

    focused_time_sec = 0.0
    non_focused_time_sec = 0.0

    frame_idx = 0
    last_pred_class = "unknown"
    last_pred_proba = 0.0

    try:
        while True:
            now = time.time()

            if P.session_duration_sec is not None and (now - start_t) >= P.session_duration_sec:
                break

            frame_bgr = cam.read()

            if now >= next_infer_t:
                pred = infer_engine.predict(frame_bgr)

                if pred.pred_proba < P.conf_threshold:
                    pred_class = "unknown"
                    pred_proba = float(pred.pred_proba)
                else:
                    pred_class = pred.pred_class
                    pred_proba = float(pred.pred_proba)

                last_pred_class, last_pred_proba = pred_class, pred_proba

                x = map_class_to_binary_or_none(pred_class)
                out = accumulator.step(x)

                if last_infer_time is not None:
                    dt = now - last_infer_time
                    if out.p >= P.threshold:
                        focused_time_sec += dt
                    else:
                        non_focused_time_sec += dt
                last_infer_time = now

                state_raw_int = -1 if x is None else int(x)

                t_sec = now - start_t
                logger.add(
                    frame_idx=frame_idx,
                    time_sec=t_sec,
                    class_raw=pred_class,
                    proba=pred_proba,
                    state_raw=state_raw_int,
                    llr=float(out.llr),
                    L=float(out.L),
                    p=float(out.p),
                    state_filtered=int(out.state_filtered),
                    focused_time_sec=float(focused_time_sec),
                    non_focused_time_sec=float(non_focused_time_sec),
                    is_unknown=1 if out.is_unknown else 0,
                )

                frame_idx += 1
                next_infer_t = now + P.frame_interval_sec

            cam.set_hud(last_pred_class, last_pred_proba)

            cam.show(frame_bgr)
            if cam.should_quit():
                break

    finally:
        cam.release()

    df = logger.to_dataframe()

    plot_paths = generate_all_plots(
        df=df,
        session_dir=session_dir,
        threshold=P.threshold,
        frame_interval_sec=P.frame_interval_sec,
        classes=classes,
    )

    print("\n[APP] Done: Results in:", session_dir)
    print("[APP] Figures:", plot_paths)


if __name__ == "__main__":
    main()
