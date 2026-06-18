import argparse
import math
from pathlib import Path
import warnings

import numpy as np


def require_dependencies():
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.signal import periodogram, spectrogram
        from scipy.stats import pearsonr
        from sklearn.feature_selection import mutual_info_regression
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from statsmodels.nonparametric.smoothers_lowess import lowess
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: {}. Install pandas matplotlib scipy scikit-learn statsmodels "
            "in the active environment before running this script.".format(exc)
        )

    return {
        "pd": pd,
        "plt": plt,
        "periodogram": periodogram,
        "spectrogram": spectrogram,
        "pearsonr": pearsonr,
        "mutual_info_regression": mutual_info_regression,
        "plot_acf": plot_acf,
        "plot_pacf": plot_pacf,
        "lowess": lowess,
        "acorr_ljungbox": acorr_ljungbox,
        "STL": STL,
        "adfuller": adfuller,
        "kpss": kpss,
    }


def build_time_index(pd, df, time_col, freq):
    if time_col and time_col in df.columns:
        time_index = pd.to_datetime(df[time_col], errors="coerce")
        if time_index.isna().any():
            raise ValueError(f"time column '{time_col}' contains invalid timestamps")
        return time_index, False

    start = "2000-01-01 00:00:00"
    freq = freq.lower() if isinstance(freq, str) else freq
    return pd.date_range(start=start, periods=len(df), freq=freq), True


def safe_kpss(kpss, series):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, lags, _ = kpss(series, regression="c", nlags="auto")
        return {"stat": float(stat), "pvalue": float(pvalue), "lags": int(lags)}
    except Exception as exc:
        return {"error": str(exc)}


def safe_adf(adfuller, series):
    try:
        stat, pvalue, lags, obs, _, _ = adfuller(series, autolag="AIC")
        return {"stat": float(stat), "pvalue": float(pvalue), "lags": int(lags), "obs": int(obs)}
    except Exception as exc:
        return {"error": str(exc)}


def ranked_periods(periodogram, series, sample_hours, top_k=8):
    freqs, power = periodogram(series, fs=1.0 / sample_hours, scaling="spectrum")
    valid = freqs > 0
    freqs = freqs[valid]
    power = power[valid]
    if len(freqs) == 0:
        return [], freqs, power

    top_idx = np.argsort(power)[::-1][:top_k]
    peaks = []
    for idx in top_idx:
        period_hours = 1.0 / freqs[idx]
        peaks.append({
            "period_hours": float(period_hours),
            "period_days": float(period_hours / 24.0),
            "power": float(power[idx]),
        })
    return peaks, freqs, power


def evaluate_nonlinear_fit(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10:
        return None

    linear_coef = np.polyfit(x, y, deg=1)
    quad_coef = np.polyfit(x, y, deg=2)

    linear_pred = np.polyval(linear_coef, x)
    quad_pred = np.polyval(quad_coef, x)

    linear_mse = float(np.mean((linear_pred - y) ** 2))
    quad_mse = float(np.mean((quad_pred - y) ** 2))
    improvement = float((linear_mse - quad_mse) / max(linear_mse, 1e-12))

    return {
        "linear_coef": linear_coef.tolist(),
        "quadratic_coef": quad_coef.tolist(),
        "linear_mse": linear_mse,
        "quadratic_mse": quad_mse,
        "quadratic_improvement_ratio": improvement,
    }


def save_time_domain_plots(env, df, time_index, target, output_dir):
    plt = env["plt"]
    STL = env["STL"]
    plot_acf = env["plot_acf"]
    plot_pacf = env["plot_pacf"]

    series = df[target].astype(float).to_numpy()
    rolling_window = min(24 * 7, max(24, len(series) // 20))
    rolling_mean = df[target].rolling(rolling_window, min_periods=1).mean()
    rolling_std = df[target].rolling(rolling_window, min_periods=1).std().fillna(0.0)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    axes[0].plot(time_index, df[target], linewidth=0.8, color="#1f77b4", label=target)
    axes[0].plot(time_index, rolling_mean, linewidth=1.5, color="#d62728", label=f"rolling mean ({rolling_window})")
    axes[0].set_title(f"{target} Time Series")
    axes[0].legend()

    axes[1].plot(time_index, rolling_std, linewidth=1.2, color="#2ca02c")
    axes[1].set_title(f"{target} Rolling Std ({rolling_window})")

    axes[2].hist(series, bins=60, color="#ff7f0e", alpha=0.85)
    axes[2].set_title(f"{target} Value Distribution")
    fig.savefig(output_dir / "time_domain_overview.png", dpi=200)
    plt.close(fig)

    stl_period = 24 if len(series) >= 48 else max(2, len(series) // 4)
    stl_result = STL(series, period=stl_period, robust=True).fit()
    fig = stl_result.plot()
    fig.set_size_inches(16, 10)
    fig.savefig(output_dir / "time_domain_stl.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    acf_lags = min(24 * 14, len(series) // 2 - 1)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    plot_acf(series, lags=acf_lags, ax=axes[0], title=f"{target} ACF")
    plot_pacf(series, lags=min(acf_lags, 24 * 7), ax=axes[1], title=f"{target} PACF", method="ywm")
    fig.savefig(output_dir / "time_domain_acf_pacf.png", dpi=200)
    plt.close(fig)

    hour_mean = df.groupby(time_index.hour)[target].mean()
    weekday_mean = df.groupby(time_index.dayofweek)[target].mean()
    month_mean = df.groupby(time_index.month)[target].mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    axes[0].plot(hour_mean.index, hour_mean.values, marker="o", color="#1f77b4")
    axes[0].set_title(f"{target} Mean by Hour of Day")
    axes[0].set_xlabel("Hour")

    axes[1].plot(weekday_mean.index, weekday_mean.values, marker="o", color="#ff7f0e")
    axes[1].set_title(f"{target} Mean by Day of Week")
    axes[1].set_xlabel("Weekday (0=Mon)")

    axes[2].plot(month_mean.index, month_mean.values, marker="o", color="#2ca02c")
    axes[2].set_title(f"{target} Mean by Month")
    axes[2].set_xlabel("Month")
    fig.savefig(output_dir / "time_domain_group_means.png", dpi=200)
    plt.close(fig)

    return {
        "rolling_window": int(rolling_window),
        "stl_period": int(stl_period),
        "hour_mean_peak_hour": int(hour_mean.idxmax()),
        "weekday_mean_peak_day": int(weekday_mean.idxmax()),
        "month_mean_peak_month": int(month_mean.idxmax()),
    }


def save_frequency_plots(env, df, target, sample_hours, output_dir):
    plt = env["plt"]
    spectrogram = env["spectrogram"]
    periodogram = env["periodogram"]

    series = df[target].astype(float).to_numpy()
    peaks, freqs, power = ranked_periods(periodogram, series, sample_hours)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    axes[0].plot(freqs, power, color="#1f77b4")
    axes[0].set_title(f"{target} Periodogram")
    axes[0].set_xlabel("Frequency (cycles/hour)")
    axes[0].set_ylabel("Power")

    periods = np.divide(1.0, freqs, out=np.zeros_like(freqs), where=freqs > 0)
    valid = np.isfinite(periods) & (periods <= 24 * 14)
    axes[1].plot(periods[valid], power[valid], color="#d62728")
    axes[1].set_title(f"{target} Periodogram by Period Length")
    axes[1].set_xlabel("Period (hours)")
    axes[1].set_ylabel("Power")
    fig.savefig(output_dir / "frequency_domain_periodogram.png", dpi=200)
    plt.close(fig)

    nperseg = min(24 * 14, len(series))
    if nperseg < 16:
        return {"top_periods": peaks, "spectrogram_saved": False}

    f, t, sxx = spectrogram(series, fs=1.0 / sample_hours, nperseg=nperseg, noverlap=nperseg // 2)
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    im = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-12), shading="gouraud", cmap="viridis")
    ax.set_title(f"{target} Spectrogram")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Frequency (cycles/hour)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    fig.savefig(output_dir / "frequency_domain_spectrogram.png", dpi=200)
    plt.close(fig)

    return {"top_periods": peaks, "spectrogram_saved": True}


def save_nonlinear_plots(env, df, target, output_dir, focus_features):
    plt = env["plt"]
    pearsonr = env["pearsonr"]
    mutual_info_regression = env["mutual_info_regression"]
    lowess = env["lowess"]

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [col for col in numeric_df.columns if col != target]
    x_all = numeric_df[feature_cols].ffill().bfill().fillna(0.0)
    y = numeric_df[target].astype(float).to_numpy()

    mi = mutual_info_regression(x_all.to_numpy(), y, random_state=2020)
    pearsons = []
    for col in feature_cols:
        corr, _ = pearsonr(x_all[col].to_numpy(), y)
        pearsons.append(corr)

    ranking = sorted(
        [
            {
                "feature": col,
                "pearson": float(corr),
                "abs_pearson": float(abs(corr)),
                "mutual_info": float(mi_value),
            }
            for col, corr, mi_value in zip(feature_cols, pearsons, mi)
        ],
        key=lambda item: item["mutual_info"],
        reverse=True
    )

    top_rank = ranking[: min(10, len(ranking))]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].barh([x["feature"] for x in top_rank][::-1], [x["mutual_info"] for x in top_rank][::-1], color="#1f77b4")
    axes[0].set_title(f"{target} Top Mutual Information Features")
    axes[0].set_xlabel("Mutual Information")

    axes[1].barh([x["feature"] for x in top_rank][::-1], [x["abs_pearson"] for x in top_rank][::-1], color="#ff7f0e")
    axes[1].set_title(f"{target} Top Absolute Pearson Correlation")
    axes[1].set_xlabel("|Pearson r|")
    fig.savefig(output_dir / "nonlinear_feature_ranking.png", dpi=200)
    plt.close(fig)

    nonlinear_summary = []
    focus = [col for col in focus_features if col in numeric_df.columns and col != target]
    if not focus:
        focus = [item["feature"] for item in ranking[:2]]

    sample_size = min(5000, len(numeric_df))
    sample_idx = np.linspace(0, len(numeric_df) - 1, sample_size, dtype=int)

    for col in focus:
        x = numeric_df[col].to_numpy()
        sample_x = x[sample_idx]
        sample_y = y[sample_idx]
        fit_info = evaluate_nonlinear_fit(sample_x, sample_y)
        if fit_info is None:
            continue

        order = np.argsort(sample_x)
        lowess_line = lowess(sample_y, sample_x, frac=0.18, return_sorted=True)

        linear_coef = np.array(fit_info["linear_coef"])
        quad_coef = np.array(fit_info["quadratic_coef"])
        linear_pred = np.polyval(linear_coef, sample_x[order])
        quad_pred = np.polyval(quad_coef, sample_x[order])

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.scatter(sample_x, sample_y, s=8, alpha=0.18, color="#1f77b4", label="samples")
        ax.plot(sample_x[order], linear_pred, color="#ff7f0e", linewidth=2, label="linear fit")
        ax.plot(sample_x[order], quad_pred, color="#2ca02c", linewidth=2, label="quadratic fit")
        ax.plot(lowess_line[:, 0], lowess_line[:, 1], color="#d62728", linewidth=2, label="LOWESS")
        ax.set_title(f"{col} vs {target}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        ax.legend()
        fig.savefig(output_dir / f"nonlinear_{col}_vs_{target}.png", dpi=200)
        plt.close(fig)

        nonlinear_summary.append({"feature": col, **fit_info})

    return {"feature_ranking": ranking, "focus_nonlinear_fits": nonlinear_summary}


def write_summary(output_dir, data_file, target, used_synthetic_time, sample_hours, temporal, frequency, nonlinear, lb, adf, kpss_result):
    lines = []
    lines.append(f"# Feature Evidence Summary: {target}")
    lines.append("")
    lines.append(f"- Data file: `{data_file}`")
    lines.append(f"- Target column: `{target}`")
    lines.append(f"- Time index source: `{'synthetic hourly index from row order' if used_synthetic_time else 'parsed timestamp column'}`")
    lines.append(f"- Sampling interval (hours): `{sample_hours}`")
    lines.append("")

    lines.append("## Temporal Evidence")
    lines.append(f"- Rolling window used: `{temporal['rolling_window']}`")
    lines.append(f"- STL seasonal period: `{temporal['stl_period']}`")
    lines.append(f"- Peak hour by group mean: `{temporal['hour_mean_peak_hour']}`")
    lines.append(f"- Peak weekday by group mean: `{temporal['weekday_mean_peak_day']}`")
    lines.append(f"- Peak month by group mean: `{temporal['month_mean_peak_month']}`")
    lines.append(f"- Ljung-Box p-value @ lag 24: `{lb['pvalue']:.6g}`")
    if "error" in adf:
        lines.append(f"- ADF failed: `{adf['error']}`")
    else:
        lines.append(f"- ADF p-value: `{adf['pvalue']:.6g}`")
    if "error" in kpss_result:
        lines.append(f"- KPSS failed: `{kpss_result['error']}`")
    else:
        lines.append(f"- KPSS p-value: `{kpss_result['pvalue']:.6g}`")
    lines.append("")

    lines.append("## Frequency Evidence")
    for idx, peak in enumerate(frequency["top_periods"], start=1):
        lines.append(
            f"- Top period {idx}: `{peak['period_hours']:.2f}` hours "
            f"(`{peak['period_days']:.2f}` days), power `{peak['power']:.6g}`"
        )
    lines.append("")

    lines.append("## Nonlinear Evidence")
    for item in nonlinear["feature_ranking"][:10]:
        lines.append(
            f"- `{item['feature']}`: mutual_info=`{item['mutual_info']:.6g}`, "
            f"pearson=`{item['pearson']:.6g}`"
        )
    lines.append("")
    for fit in nonlinear["focus_nonlinear_fits"]:
        lines.append(
            f"- `{fit['feature']}` quadratic improvement ratio over linear fit: "
            f"`{fit['quadratic_improvement_ratio']:.4%}`"
        )
    lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze temporal, frequency, and nonlinear evidence in the dataset.")
    parser.add_argument("--data", type=str, default="./data/dataset_input_jiuzheng.csv")
    parser.add_argument("--target", type=str, default="CHWTON", help="target column to analyze, e.g. CHWTON/KW/HTmmBTU")
    parser.add_argument("--time_col", type=str, default=None, help="optional timestamp column name")
    parser.add_argument("--freq", type=str, default="H", help="sampling frequency used when no timestamp column exists")
    parser.add_argument("--sample_hours", type=float, default=1.0, help="hours per sample")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--focus_features",
        nargs="*",
        default=["temperature", "wet_bulb_temperature", "dew_point_temperature"],
        help="features used for nonlinear scatter and fit plots"
    )
    args = parser.parse_args()

    env = require_dependencies()
    pd = env["pd"]
    acorr_ljungbox = env["acorr_ljungbox"]
    adfuller = env["adfuller"]
    kpss = env["kpss"]

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise SystemExit(f"target column not found: {args.target}")

    time_index, used_synthetic_time = build_time_index(pd, df, args.time_col, args.freq)

    output_dir = Path(args.output_dir) if args.output_dir else Path("./output") / f"feature_evidence_{args.target}"
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal = save_time_domain_plots(env, df, time_index, args.target, output_dir)
    frequency = save_frequency_plots(env, df, args.target, args.sample_hours, output_dir)
    nonlinear = save_nonlinear_plots(env, df, args.target, output_dir, args.focus_features)

    series = df[args.target].astype(float).to_numpy()
    lb_df = acorr_ljungbox(series, lags=[24], return_df=True)
    lb = {
        "stat": float(lb_df["lb_stat"].iloc[0]),
        "pvalue": float(lb_df["lb_pvalue"].iloc[0]),
    }
    adf = safe_adf(adfuller, series)
    kpss_result = safe_kpss(kpss, series)

    write_summary(
        output_dir=output_dir,
        data_file=str(data_path),
        target=args.target,
        used_synthetic_time=used_synthetic_time,
        sample_hours=args.sample_hours,
        temporal=temporal,
        frequency=frequency,
        nonlinear=nonlinear,
        lb=lb,
        adf=adf,
        kpss_result=kpss_result,
    )

    print(f"analysis saved to directory: {output_dir.resolve()}")
    print(f"summary file: {(output_dir / 'summary.md').resolve()}")


if __name__ == "__main__":
    main()
