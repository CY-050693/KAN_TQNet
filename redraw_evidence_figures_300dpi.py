from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset_input_jiuzheng.csv"
SAVE_DIR = ROOT / "paper_figures_300dpi" / "evidence_figures_redrawn_300dpi"
DPI = 300

LOADS = [
    ("KW", "Electricity", "#0b3c5d"),
    ("CHWTON", "Cooling", "#d95f02"),
    ("HTmmBTU", "Heating", "#1b9e77"),
]

NONLINEAR_PANELS = [
    ("temperature", "KW", "Temperature vs Electricity", "#0b3c5d"),
    ("wet_bulb_temperature", "CHWTON", "Wet-Bulb vs Cooling", "#d95f02"),
    ("dew_point_temperature", "HTmmBTU", "Dew Point vs Heating", "#1b9e77"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d1d5db",
            "grid.alpha": 0.30,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92, "pad": 2.0},
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def autocorr(series: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    values = [1.0]
    for lag in range(1, nlags + 1):
        values.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(values)


def load_dataset() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    df = pd.read_csv(DATA_PATH)
    time_index = pd.date_range("2000-01-01 00:00:00", periods=len(df), freq="H")
    return df, time_index


def smooth_curve(x: np.ndarray, y: np.ndarray, num: int = 300) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs = np.asarray(x[order], dtype=float)
    ys = np.asarray(y[order], dtype=float)
    x_grid = np.linspace(xs.min(), xs.max(), num)
    y_interp = np.interp(x_grid, xs, ys)
    y_smooth = gaussian_filter1d(y_interp, sigma=6)
    return x_grid, y_smooth


def plot_temporal_evidence(df: pd.DataFrame, time_index: pd.DatetimeIndex) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))

    ax = axes[0, 0]
    for col, label, color in LOADS:
        series = df[col].astype(float)
        normalized = (series - series.mean()) / series.std()
        smooth = normalized.rolling(24, min_periods=1).mean()
        ax.plot(time_index, smooth, color=color, linewidth=1.25, label=label)
    ax.set_title("24-hour Smoothed Standardized Load Series")
    ax.set_ylabel("Standardized load")
    ax.set_xlim(time_index[0], time_index[-1])
    date_ticks = pd.date_range(time_index[0].normalize(), time_index[-1].normalize(), freq="6MS")
    ax.set_xticks(date_ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=0, pad=3)
    ax.margins(x=0)
    ax.legend(loc="upper left", ncol=3)
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    hour = time_index.hour
    for col, label, color in LOADS:
        profile = df.groupby(hour)[col].mean()
        ax.plot(profile.index, profile.values, marker="o", markersize=3.2, linewidth=1.8, color=color, label=label)
    ax.set_title("Average Daily Profile")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Load")
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    weekday = time_index.dayofweek
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for col, label, color in LOADS:
        profile = df.groupby(weekday)[col].mean()
        ax.plot(profile.index, profile.values, marker="o", markersize=3.2, linewidth=1.8, color=color, label=label)
    ax.set_xticks(range(7), weekday_labels)
    ax.set_title("Average Weekly Profile")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Load")
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    nlags = 24 * 7
    lags = np.arange(nlags + 1)
    for col, label, color in LOADS:
        acf = autocorr(df[col].astype(float).to_numpy(), nlags)
        ax.plot(lags, acf, linewidth=1.8, color=color, label=label)
    ax.axvline(24, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.axvline(168, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.text(24, 0.92, "24h", color="#4b5563", fontsize=9, ha="center")
    ax.text(168, 0.92, "168h", color="#4b5563", fontsize=9, ha="center")
    ax.set_title("Autocorrelation Structure")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    add_panel_label(ax, "D")

    fig.suptitle("Temporal Evidence of Multi-Energy Load Dynamics", y=0.98, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.91, wspace=0.22, hspace=0.28)
    return save_figure(fig, "temporal_evidence_redrawn_300dpi.png")


def plot_frequency_evidence(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1))

    ax = axes[0]
    for col, label, color in LOADS:
        series = df[col].astype(float).to_numpy()
        freqs, power = periodogram(series, fs=1.0, scaling="spectrum")
        valid = freqs > 0
        periods = 1.0 / freqs[valid]
        power = power[valid]
        valid_period = (periods >= 2) & (periods <= 24 * 365)
        periods = periods[valid_period]
        power = power[valid_period]
        power = power / np.max(power)
        ax.plot(periods, power, color=color, linewidth=1.6, label=label)
    ax.set_xscale("log")
    ax.axvline(24, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.axvline(168, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_title("Normalized Periodogram")
    ax.set_xlabel("Period (hours, log scale)")
    ax.set_ylabel("Normalized power")
    ax.legend(loc="upper right")
    add_panel_label(ax, "A")

    ax = axes[1]
    focus_periods = np.array([24, 168, 24 * 30, 24 * 180, 24 * 330], dtype=float)
    x = np.arange(len(focus_periods))
    width = 0.22
    for idx, (col, label, color) in enumerate(LOADS):
        series = df[col].astype(float).to_numpy()
        freqs, power = periodogram(series, fs=1.0, scaling="spectrum")
        valid = freqs > 0
        periods = 1.0 / freqs[valid]
        power = power[valid]
        strengths = []
        for p in focus_periods:
            j = int(np.argmin(np.abs(periods - p)))
            strengths.append(power[j])
        strengths = np.asarray(strengths, dtype=float)
        strengths = strengths / np.max(strengths)
        ax.bar(x + (idx - 1) * width, strengths, width=width, color=color, alpha=0.90, label=label)
    ax.set_xticks(x, ["24h", "168h", "30d", "180d", "330d"])
    ax.set_title("Power at Representative Periods")
    ax.set_ylabel("Relative power")
    add_panel_label(ax, "B")

    fig.suptitle("Frequency Evidence of Multi-Scale Periodicity", y=0.98, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.88, wspace=0.18)
    return save_figure(fig, "frequency_evidence_redrawn_300dpi.png")


def plot_nonlinear_evidence(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    target = "KW"
    feature_cols = [c for c in numeric_df.columns if c != target]
    x_all = numeric_df[feature_cols].ffill().bfill().fillna(0.0)
    y = numeric_df[target].astype(float).to_numpy()
    mi = mutual_info_regression(x_all.to_numpy(), y, random_state=2020)
    corr = np.array([abs(pearsonr(x_all[col].to_numpy(), y)[0]) for col in feature_cols])
    ranking = pd.DataFrame({"feature": feature_cols, "mutual_info": mi, "abs_pearson": corr}).sort_values(
        "mutual_info", ascending=False
    )
    top_rank = ranking.head(8).iloc[::-1]

    ax = axes[0, 0]
    scatter = ax.scatter(
        top_rank["abs_pearson"],
        top_rank["mutual_info"],
        s=85,
        c=np.linspace(0.15, 0.85, len(top_rank)),
        cmap="viridis",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
    )
    for _, row in top_rank.iterrows():
        ax.text(row["abs_pearson"] + 0.008, row["mutual_info"] + 0.01, row["feature"], fontsize=8.5)
    ax.set_title("Feature Dependence Ranking")
    ax.set_xlabel("|Pearson r|")
    ax.set_ylabel("Mutual information")
    add_panel_label(ax, "A")

    for ax, (feature, target_col, title, color), label in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        NONLINEAR_PANELS,
        ["B", "C", "D"],
    ):
        x = df[feature].astype(float).to_numpy()
        y = df[target_col].astype(float).to_numpy()
        sample_size = min(4500, len(df))
        sample_idx = np.linspace(0, len(df) - 1, sample_size, dtype=int)
        sx = x[sample_idx]
        sy = y[sample_idx]
        order = np.argsort(sx)
        linear_coef = np.polyfit(sx, sy, deg=1)
        linear_pred = np.polyval(linear_coef, sx[order])
        smooth_x, smooth_y = smooth_curve(sx, sy)

        ax.hexbin(sx, sy, gridsize=34, cmap="Blues", mincnt=1, linewidths=0.0)
        ax.plot(sx[order], linear_pred, color="#ef4444", linewidth=1.6, label="Linear fit")
        ax.plot(smooth_x, smooth_y, color=color, linewidth=2.2, label="Smoothed trend")
        ax.set_title(title)
        ax.set_xlabel(feature.replace("_", " "))
        ax.set_ylabel(target_col)
        ax.legend(loc="upper left")
        add_panel_label(ax, label)

    fig.suptitle("Nonlinear Evidence of Load-Feature Coupling", y=0.98, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.91, wspace=0.22, hspace=0.28)
    return save_figure(fig, "nonlinear_evidence_redrawn_300dpi.png")


def plot_motivation_triptych(df: pd.DataFrame, time_index: pd.DatetimeIndex) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    ax = axes[0]
    for col, label, color in LOADS:
        smooth = df[col].astype(float).rolling(24, min_periods=1).mean()
        normalized = (smooth - smooth.mean()) / smooth.std()
        ax.plot(time_index, normalized, color=color, linewidth=1.15, label=label)
    ax.set_title("Temporal Dependency")
    ax.set_ylabel("Standardized load")
    ax.legend(loc="upper left", ncol=1)
    add_panel_label(ax, "A")

    ax = axes[1]
    for col, label, color in LOADS:
        series = df[col].astype(float).to_numpy()
        freqs, power = periodogram(series, fs=1.0, scaling="spectrum")
        valid = freqs > 0
        periods = 1.0 / freqs[valid]
        power = power[valid]
        valid_period = (periods >= 2) & (periods <= 24 * 365)
        periods = periods[valid_period]
        power = power[valid_period]
        ax.plot(periods, power / np.max(power), color=color, linewidth=1.35)
    ax.set_xscale("log")
    ax.axvline(24, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.axvline(168, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_title("Periodic Structure")
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("Normalized power")
    add_panel_label(ax, "B")

    ax = axes[2]
    x = df["wet_bulb_temperature"].astype(float).to_numpy()
    y = df["CHWTON"].astype(float).to_numpy()
    sample_idx = np.linspace(0, len(df) - 1, min(4500, len(df)), dtype=int)
    sx = x[sample_idx]
    sy = y[sample_idx]
    smooth_x, smooth_y = smooth_curve(sx, sy)
    ax.hexbin(sx, sy, gridsize=34, cmap="Oranges", mincnt=1, linewidths=0.0)
    ax.plot(smooth_x, smooth_y, color="#b45309", linewidth=2.2)
    ax.set_title("Nonlinear Coupling")
    ax.set_xlabel("wet bulb temperature")
    ax.set_ylabel("CHWTON")
    add_panel_label(ax, "C")

    fig.suptitle("Motivation for Temporal, Spectral, and Nonlinear Modeling", y=0.98, fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.84, wspace=0.28)
    return save_figure(fig, "motivation_triptych_redrawn_300dpi.png")


def main() -> None:
    configure_style()
    df, time_index = load_dataset()
    outputs = [
        plot_temporal_evidence(df, time_index),
        plot_frequency_evidence(df),
        plot_nonlinear_evidence(df),
        plot_motivation_triptych(df, time_index),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
