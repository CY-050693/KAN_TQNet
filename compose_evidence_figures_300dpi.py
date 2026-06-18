from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "paper_figures_300dpi" / "evidence_figures_300dpi"
DPI = 300


def load_and_crop_image(img_path: Path, white_threshold: float = 0.985, pad: int = 12):
    img = mpimg.imread(img_path)
    if img.dtype.kind in {"u", "i"}:
        img = img.astype(np.float32) / 255.0

    if img.ndim == 2:
        mask = img < white_threshold
    else:
        rgb = img[..., :3]
        mask = np.any(rgb < white_threshold, axis=-1)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0] - 1, y1 + pad)
    x1 = min(img.shape[1] - 1, x1 + pad)
    return img[y0 : y1 + 1, x0 : x1 + 1]


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92, "pad": 2.0},
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def compose_temporal_evidence() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), facecolor="white")
    panels = [
        ("A", ROOT / "time_domain_overview.png", "Time Series Overview"),
        ("B", ROOT / "time_domain_stl.png", "STL Decomposition"),
        ("C", ROOT / "time_domain_acf_pacf.png", "ACF and PACF"),
        ("D", ROOT / "time_domain_group_means.png", "Grouped Mean Profiles"),
    ]

    for ax, (label, img_path, title) in zip(axes.flat, panels):
        img = load_and_crop_image(img_path)
        ax.imshow(img)
        ax.set_title(title, pad=6, fontsize=12, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, label)

    fig.suptitle("Temporal Evidence of Multi-Energy Load Series", y=0.965, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.04, top=0.91, wspace=0.05, hspace=0.12)
    return save_figure(fig, "temporal_evidence_300dpi.png")


def compose_frequency_evidence() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), facecolor="white")
    panels = [
        ("A", ROOT / "frequency_domain_periodogram.png", "Periodogram"),
        ("B", ROOT / "frequency_domain_spectrogram.png", "Spectrogram"),
    ]

    for ax, (label, img_path, title) in zip(axes.flat, panels):
        img = load_and_crop_image(img_path)
        ax.imshow(img)
        ax.set_title(title, pad=6, fontsize=12, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, label)

    fig.suptitle("Frequency-Domain Evidence of Periodic Structure", y=0.955, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.06, top=0.87, wspace=0.045)
    return save_figure(fig, "frequency_evidence_300dpi.png")


def compose_nonlinear_evidence() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), facecolor="white")
    panels = [
        ("A", ROOT / "nonlinear_feature_ranking.png", "Feature Ranking"),
        ("B", ROOT / "nonlinear_temperature_vs_KW.png", "Temperature vs Electricity"),
        ("C", ROOT / "nonlinear_temperature_vs_CHWTON.png", "Temperature vs Cooling"),
        ("D", ROOT / "nonlinear_temperature_vs_HTmmBTU.png", "Temperature vs Heating"),
    ]

    for ax, (label, img_path, title) in zip(axes.flat, panels):
        img = load_and_crop_image(img_path)
        ax.imshow(img)
        ax.set_title(title, pad=6, fontsize=12, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, label)

    fig.suptitle("Nonlinear Evidence of Load-Feature Coupling", y=0.965, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.04, top=0.91, wspace=0.05, hspace=0.12)
    return save_figure(fig, "nonlinear_evidence_300dpi.png")


def compose_triptych_summary() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), facecolor="white")
    panels = [
        ("A", ROOT / "time_domain_overview.png", "Temporal Dependency"),
        ("B", ROOT / "frequency_domain_periodogram.png", "Periodic Structure"),
        ("C", ROOT / "nonlinear_feature_ranking.png", "Nonlinear Coupling"),
    ]

    for ax, (label, img_path, title) in zip(axes.flat, panels):
        img = load_and_crop_image(img_path)
        ax.imshow(img)
        ax.set_title(title, pad=6, fontsize=12.5, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, label)

    fig.suptitle("Motivation for Temporal, Spectral, and Nonlinear Modeling", y=0.93, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.83, wspace=0.04)
    return save_figure(fig, "motivation_triptych_300dpi.png")


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    outputs = [
        compose_temporal_evidence(),
        compose_frequency_evidence(),
        compose_nonlinear_evidence(),
        compose_triptych_summary(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
