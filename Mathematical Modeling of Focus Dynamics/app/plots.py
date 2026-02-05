from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_time_axis(df: pd.DataFrame) -> np.ndarray:
    if "time_sec" not in df.columns:
        raise ValueError("time_sec column is required.")
    return df["time_sec"].astype(float).to_numpy()


def _compute_dt_from_time_sec(df: pd.DataFrame) -> np.ndarray:
    t = _ensure_time_axis(df)
    if len(t) == 0:
        return np.array([])
    dt = np.zeros_like(t)
    dt[1:] = np.maximum(t[1:] - t[:-1], 0.0)
    return dt


def _ewma(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if len(x) == 0:
        return y
    y[0] = float(x[0])
    for i in range(1, len(x)):
        y[i] = alpha * float(x[i]) + (1 - alpha) * y[i - 1]
    return y


def _canonical_4classes(classes: List[str]) -> List[str]:
    canonical = ["focused_writing", "focused_reading", "not_phone", "not_activity"]
    present = [c for c in canonical if c in classes]
    return present if len(present) == 4 else canonical


def _pick_time_unit(total_sec: float) -> Tuple[str, float]:
    total_sec = float(max(total_sec, 0.0))
    if total_sec < 5 * 60:
        return "seconds", 1.0
    if total_sec < 60 * 60:
        return "minutes", 60.0
    return "hours", 3600.0


def _format_duration(seconds: float, unit: str) -> str:
    s = float(max(seconds, 0.0))
    if unit == "seconds":
        return f"{s:.0f} s"
    if unit == "minutes":
        return f"{(s / 60.0):.1f} min"
    if unit == "hours":
        return f"{(s / 3600.0):.2f} h"
    return f"{s:.1f}"


def _time_axis(df: pd.DataFrame) -> Tuple[np.ndarray, str, float]:
    t_sec = _ensure_time_axis(df)
    total_sec = float(t_sec[-1]) if len(t_sec) else 0.0
    unit, scale = _pick_time_unit(total_sec)
    t_disp = t_sec / scale if scale > 0 else t_sec
    return t_disp, unit, scale


def plot_concentration(df: pd.DataFrame, out_path: Path, threshold: float):
    if "p" not in df.columns:
        raise ValueError("Column 'p' missing.")
    if "focused_time_sec" not in df.columns or "non_focused_time_sec" not in df.columns:
        raise ValueError("Columns 'focused_time_sec' and 'non_focused_time_sec' are required.")

    t, unit, _scale = _time_axis(df)
    p = df["p"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, p, color="black", linewidth=2)
    ax.axhline(threshold, linestyle="--", color="gray", linewidth=1)

    focused_color = "#a8ddb5"
    not_focused_color = "#f4a6a6"

    ax.fill_between(
        t, p, threshold,
        where=(p >= threshold),
        interpolate=True,
        color=focused_color,
        alpha=0.45,
        label="Focused",
    )
    ax.fill_between(
        t, p, threshold,
        where=(p < threshold),
        interpolate=True,
        color=not_focused_color,
        alpha=0.40,
        label="Not focused",
    )

    ax.set_title("Focus level over time")
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Focus level")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_activity_curves_4panels(
    df: pd.DataFrame,
    out_path: Path,
    classes: List[str],
    smoothing: str = "ewma",
    alpha: float = 0.2,
    window: int = 5,
):
    if "class_raw" not in df.columns:
        raise ValueError("Column 'class_raw' missing.")

    t, unit, _scale = _time_axis(df)
    y_class = df["class_raw"].astype(str).to_numpy()
    ordered = _canonical_4classes(classes)

    class_display_names = {
        "focused_writing": "Writing Attention",
        "focused_reading": "Reading Attention",
        "not_phone": "Digital Distraction",
        "not_activity": "Cognitive Inactivity",
    }

    color_map = {
        "focused_writing": "#1f77b4",
        "focused_reading": "#2ca02c",
        "not_phone": "#d62728",
        "not_activity": "#7f7f7f",
    }

    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    for ax, cls in zip(axes, ordered):
        x = (y_class == cls).astype(float)

        if smoothing == "ewma":
            xs = _ewma(x, alpha=alpha)
        else:
            w = max(int(window), 1)
            kernel = np.ones(w) / w
            xs = np.convolve(x, kernel, mode="same")

        ax.plot(t, xs, color=color_map.get(cls, "black"), linewidth=2)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.set_title(class_display_names.get(cls, cls), fontsize=11, loc="left")
        ax.set_ylabel("")

    fig.suptitle("Activity dominance over time", fontsize=14)
    fig.text(0.04, 0.5, "Activity dominance", va="center", rotation="vertical", fontsize=12)
    axes[-1].set_xlabel(f"Time ({unit})")

    fig.tight_layout(rect=[0.06, 0.03, 1.0, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_barchart_time_per_class_coherent(df: pd.DataFrame, out_path: Path, classes: List[str]):
    if "class_raw" not in df.columns:
        raise ValueError("Column 'class_raw' missing.")

    t_sec = _ensure_time_axis(df)
    total_sec = float(t_sec[-1]) if len(t_sec) else 0.0
    unit, scale = _pick_time_unit(total_sec)

    dt = _compute_dt_from_time_sec(df)
    y_class = df["class_raw"].astype(str).to_numpy()
    ordered = _canonical_4classes(classes)

    time_per_class_sec = {c: 0.0 for c in ordered}
    for cls, dti in zip(y_class, dt):
        if cls in time_per_class_sec:
            time_per_class_sec[cls] += float(dti)

    display_names = {
        "focused_writing": "Writing Attention",
        "focused_reading": "Reading Attention",
        "not_phone": "Digital Distraction",
        "not_activity": "Cognitive Inactivity",
    }

    x_labels = [display_names.get(c, c) for c in ordered]
    values = [time_per_class_sec[c] / scale for c in ordered]

    color_map = {
        "focused_writing": "#1f77b4",
        "focused_reading": "#2ca02c",
        "not_phone": "#d62728",
        "not_activity": "#7f7f7f",
    }
    colors = [color_map.get(c, "gray") for c in ordered]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x_labels, values, color=colors)
    ax.set_title("Time spent in different cognitive activity states")
    ax.set_ylabel(f"Time spent ({unit})")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_donut_concentration(df: pd.DataFrame, out_path: Path):
    if "focused_time_sec" not in df.columns or "non_focused_time_sec" not in df.columns:
        raise ValueError("Columns 'focused_time_sec' and 'non_focused_time_sec' are required.")

    focused_time = float(df["focused_time_sec"].iloc[-1]) if len(df) else 0.0
    non_focused_time = float(df["non_focused_time_sec"].iloc[-1]) if len(df) else 0.0
    total = max(focused_time + non_focused_time, 1e-6)

    unit, _scale = _pick_time_unit(total)
    focused_pct = 100.0 * focused_time / total

    sizes = [focused_time, non_focused_time]
    labels = ["Focused", "Not focused"]
    colors = ["#a8ddb5", "#f4a6a6"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(
        sizes,
        labels=None,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.35, edgecolor="white"),
    )

    ax.text(0, 0, f"{focused_pct:.0f}%", ha="center", va="center", fontsize=28, fontweight="bold")

    def _annotate_wedge_time(wedge, text, dy=0.0):
        ang = 0.5 * (wedge.theta1 + wedge.theta2)
        ang_rad = np.deg2rad(ang)
        x = np.cos(ang_rad) * 1.0
        y = np.sin(ang_rad) * 1.0
        xt = np.cos(ang_rad) * 1.25
        yt = np.sin(ang_rad) * 1.25 + dy
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(xt, yt),
            ha="center",
            va="center",
            fontsize=14,
            arrowprops=dict(arrowstyle="-", color="gray", lw=2),
        )

    _annotate_wedge_time(wedges[0], f" {_format_duration(focused_time, unit)}")
    _annotate_wedge_time(wedges[1], f" {_format_duration(non_focused_time, unit)}")

    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
    ax.set_title("Temporal Distribution of Cognitive Focus")
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    df: pd.DataFrame,
    session_dir: Path,
    threshold: float,
    frame_interval_sec: float,
    classes: List[str],
) -> Dict[str, Path]:
    session_dir.mkdir(parents=True, exist_ok=True)

    p1 = session_dir / "fig_concentration.png"
    p2 = session_dir / "fig_activity_curves.png"
    p3 = session_dir / "fig_barchart.png"
    p4 = session_dir / "fig_donut.png"

    plot_concentration(df, p1, threshold=threshold)
    plot_activity_curves_4panels(df, p2, classes=classes, smoothing="ewma", alpha=0.2, window=5)
    plot_barchart_time_per_class_coherent(df, p3, classes=classes)
    plot_donut_concentration(df, p4)

    return {
        "concentration": p1,
        "activity_curves": p2,
        "barchart": p3,
        "donut": p4,
    }
