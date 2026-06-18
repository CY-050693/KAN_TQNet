from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
SAVE_DIR = ROOT / "paper_figures_300dpi" / "KAN_TQNet_prediction_error_300dpi"
DPI = 300
CURVE_POINTS = 336

HORIZONS = [24, 48, 72, 96]
CHANNELS = [
    ("Electricity", 0, "#0b3c5d"),
    ("Cooling", 1, "#d95f02"),
    ("Heating", 2, "#1b9e77"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def load_first_step_predictions(horizon: int) -> tuple[np.ndarray, np.ndarray]:
    run_dir = OUTPUT_ROOT / f"KAN_TQNet_{horizon}_168" / "result"
    y_true = np.load(run_dir / "all_y_true.npy")
    y_pred = np.load(run_dir / "all_predict_value.npy")

    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError(f"Unexpected tensor shape for horizon={horizon}: {y_true.shape}, {y_pred.shape}")

    return y_true[:, 0, :], y_pred[:, 0, :]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def save_figure(fig: plt.Figure, filename: str) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_DIR / filename, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_prediction_curve_figure(horizon: int, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    n = min(CURVE_POINTS, len(y_true))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for ax, (channel_name, channel_idx, color) in zip(axes, CHANNELS):
        actual = y_true[:n, channel_idx]
        pred = y_pred[:n, channel_idx]
        ax.plot(actual, color="#111827", linewidth=1.35, label="Actual")
        ax.plot(pred, color=color, linewidth=1.20, alpha=0.95, label="Prediction")
        ax.set_ylabel("Load")
        ax.set_title(
            f"{channel_name} | Horizon {horizon} | MAE={mae(actual, pred):.2f}, RMSE={rmse(actual, pred):.2f}",
            loc="left",
            pad=6,
        )
        ax.legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("Sample index (first-step forecast sequence)")
    fig.suptitle(f"KAN_TQNet Prediction vs Actual Curves ({horizon}-step horizon)", y=0.995)
    save_figure(fig, f"KAN_TQNet_prediction_vs_actual_{horizon}step_300dpi.png")


def plot_error_distribution_figure(horizon: int, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    for ax, (channel_name, channel_idx, color) in zip(axes, CHANNELS):
        error = y_pred[:, channel_idx] - y_true[:, channel_idx]
        err_mae = float(np.mean(np.abs(error)))
        err_mean = float(np.mean(error))
        err_std = float(np.std(error))

        ax.hist(error, bins=45, color=color, alpha=0.72, edgecolor="white")
        ax.axvline(0.0, color="#111827", linewidth=1.2, linestyle="--", label="Zero error")
        ax.axvline(err_mean, color="#b91c1c", linewidth=1.2, linestyle="-", label="Mean error")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{channel_name} | Horizon {horizon} | Mean={err_mean:.2f}, Std={err_std:.2f}, MAE={err_mae:.2f}",
            loc="left",
            pad=6,
        )
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Prediction error (prediction - actual)")
    fig.suptitle(f"KAN_TQNet Error Distribution ({horizon}-step horizon, first-step forecast)", y=0.995)
    save_figure(fig, f"KAN_TQNet_error_distribution_{horizon}step_300dpi.png")


def plot_combined_prediction_curves(results: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(len(HORIZONS), len(CHANNELS), figsize=(16, 14), sharex=False)

    for row_idx, horizon in enumerate(HORIZONS):
        y_true, y_pred = results[horizon]
        n = min(CURVE_POINTS, len(y_true))
        for col_idx, (channel_name, channel_idx, color) in enumerate(CHANNELS):
            ax = axes[row_idx, col_idx]
            actual = y_true[:n, channel_idx]
            pred = y_pred[:n, channel_idx]
            ax.plot(actual, color="#111827", linewidth=1.10, label="Actual")
            ax.plot(pred, color=color, linewidth=1.00, alpha=0.95, label="Prediction")
            ax.set_title(f"{channel_name} | H={horizon} | MAE={mae(actual, pred):.1f}", pad=5)
            if row_idx == len(HORIZONS) - 1:
                ax.set_xlabel("Sample index")
            if col_idx == 0:
                ax.set_ylabel("Load")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", ncol=2)

    fig.suptitle("KAN_TQNet Prediction vs Actual Curves Across Horizons", y=0.995)
    save_figure(fig, "KAN_TQNet_prediction_vs_actual_all_horizons_300dpi.png")


def plot_combined_error_distributions(results: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(len(HORIZONS), len(CHANNELS), figsize=(16, 14), sharex=False)

    for row_idx, horizon in enumerate(HORIZONS):
        y_true, y_pred = results[horizon]
        for col_idx, (channel_name, channel_idx, color) in enumerate(CHANNELS):
            ax = axes[row_idx, col_idx]
            error = y_pred[:, channel_idx] - y_true[:, channel_idx]
            ax.hist(error, bins=40, color=color, alpha=0.72, edgecolor="white")
            ax.axvline(0.0, color="#111827", linewidth=1.1, linestyle="--")
            ax.axvline(float(np.mean(error)), color="#b91c1c", linewidth=1.1)
            ax.set_title(f"{channel_name} | H={horizon} | MAE={np.mean(np.abs(error)):.1f}", pad=5)
            if row_idx == len(HORIZONS) - 1:
                ax.set_xlabel("Error")
            if col_idx == 0:
                ax.set_ylabel("Count")

    fig.suptitle("KAN_TQNet Error Distributions Across Horizons", y=0.995)
    save_figure(fig, "KAN_TQNet_error_distribution_all_horizons_300dpi.png")


def write_summary(results: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    lines = [
        "KAN_TQNet first-step forecast plotting summary",
        "Output resolution: 300 dpi",
        f"Output directory: {SAVE_DIR}",
        "",
    ]

    for horizon in HORIZONS:
        y_true, y_pred = results[horizon]
        lines.append(f"[Horizon {horizon}]")
        for channel_name, channel_idx, _ in CHANNELS:
            actual = y_true[:, channel_idx]
            pred = y_pred[:, channel_idx]
            error = pred - actual
            lines.append(
                f"{channel_name}: MAE={np.mean(np.abs(error)):.4f}, "
                f"RMSE={np.sqrt(np.mean(error ** 2)):.4f}, "
                f"MeanError={np.mean(error):.4f}, StdError={np.std(error):.4f}"
            )
        lines.append("")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for horizon in HORIZONS:
        y_true, y_pred = load_first_step_predictions(horizon)
        results[horizon] = (y_true, y_pred)
        plot_prediction_curve_figure(horizon, y_true, y_pred)
        plot_error_distribution_figure(horizon, y_true, y_pred)

    plot_combined_prediction_curves(results)
    plot_combined_error_distributions(results)
    write_summary(results)
    print(f"Saved figures to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
