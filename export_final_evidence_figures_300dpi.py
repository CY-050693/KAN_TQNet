from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REDRAW_DIR = ROOT / "paper_figures_300dpi" / "evidence_figures_redrawn_300dpi"
SAVE_DIR = ROOT / "paper_figures_300dpi" / "final_evidence_figures_300dpi"
DPI = 300


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92, "pad": 2.0},
    )


def crop_embedded_fig_banner(img, crop_ratio: float = 0.075):
    top = max(1, int(img.shape[0] * crop_ratio))
    return img[top:, ...]


def save_single_image(src: Path, title: str, filename: str) -> Path:
    img = crop_embedded_fig_banner(mpimg.imread(src))
    fig, ax = plt.subplots(figsize=(12.6, 8.2), facecolor="white")
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(title, y=0.965, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.92)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def save_frequency(src: Path, filename: str) -> Path:
    img = crop_embedded_fig_banner(mpimg.imread(src))
    fig, ax = plt.subplots(figsize=(12.8, 5.0), facecolor="white")
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("Frequency-Domain Evidence of Multi-Scale Periodicity", y=0.97, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.88)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    outputs = [
        save_single_image(
            REDRAW_DIR / "nonlinear_evidence_redrawn_300dpi.png",
            "Nonlinear Evidence of Load-Feature Coupling",
            "Fig1_nonlinear_evidence_300dpi.png",
        ),
        save_single_image(
            REDRAW_DIR / "temporal_evidence_redrawn_300dpi.png",
            "Temporal Evidence of Multi-Energy Load Dynamics",
            "Fig2_temporal_evidence_300dpi.png",
        ),
        save_frequency(
            REDRAW_DIR / "frequency_evidence_redrawn_300dpi.png",
            "Fig3_frequency_evidence_300dpi.png",
        ),
    ]

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
