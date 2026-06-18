from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
HORIZONS = [24, 48, 72, 96]
CHANNELS = ["electricity", "cooling", "heating"]

EXPERIMENTS = [
    ("core_full", "完整模型"),
    ("core_no_kan", "去掉 KAN 主干"),
    ("core_no_tq", "去掉 Temporal Query"),
    ("core_no_freq", "去掉频域分支"),
    ("core_no_trend", "去掉趋势残差分解"),
    ("core_no_hda", "去掉 HDA 路由"),
    ("core_no_timemixer", "去掉 TimeMixer 精修"),
]

SOURCE_TO_PREFIX = {
    "overall_best": "final_overall_best",
    "balanced_best": "final_balanced_best",
    "hybrid_best": "final_hybrid_best",
}

METRIC_RE = re.compile(
    r"^(final_(?:overall|balanced|hybrid)_best)"
    r"(?:_(electricity|cooling|heating))?"
    r" mae ([0-9.]+) \| mape ([0-9.]+) \| corr ([0-9.]+) \| rmse ([0-9.]+)$"
)
SOURCE_RE = re.compile(r"^final export source: ([a-z_]+)$")


def load_metrics(log_path: Path) -> tuple[str, dict[str, dict[str, dict[str, float]]]]:
    export_source = ""
    metrics: dict[str, dict[str, dict[str, float]]] = {}

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        source_match = SOURCE_RE.match(line)
        if source_match:
            export_source = source_match.group(1)
            continue

        metric_match = METRIC_RE.match(line)
        if not metric_match:
            continue

        prefix, channel, mae, mape, corr, rmse = metric_match.groups()
        target = channel or "overall"
        metrics.setdefault(prefix, {})[target] = {
            "mae": float(mae),
            "mape": float(mape),
            "corr": float(corr),
            "rmse": float(rmse),
        }

    if not export_source:
        raise ValueError(f"Missing final export source in {log_path}")

    return export_source, metrics


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        for tag, desc in EXPERIMENTS:
            output_dir = OUTPUT_ROOT / f"KAN_TQNet_{horizon}_168_{tag}_s0"
            log_path = output_dir / "result" / "data.txt"
            if not log_path.exists():
                raise FileNotFoundError(f"Result log not found: {log_path}")

            export_source, metrics = load_metrics(log_path)
            selected_prefix = SOURCE_TO_PREFIX.get(export_source)
            if not selected_prefix or selected_prefix not in metrics:
                raise ValueError(f"Unsupported export source {export_source} in {log_path}")

            selected = metrics[selected_prefix]
            overall_best = metrics.get("final_overall_best", {})

            row: dict[str, object] = {
                "horizon": horizon,
                "experiment_tag": tag,
                "description": desc,
                "output_dir": output_dir.name,
                "final_export_source": export_source,
                "selected_prefix": selected_prefix,
                "selected_overall_mae": selected["overall"]["mae"],
                "selected_overall_rmse": selected["overall"]["rmse"],
                "selected_overall_mape": selected["overall"]["mape"],
                "selected_overall_corr": selected["overall"]["corr"],
                "overall_best_mae": overall_best.get("overall", {}).get("mae", float("nan")),
                "overall_best_rmse": overall_best.get("overall", {}).get("rmse", float("nan")),
                "overall_best_mape": overall_best.get("overall", {}).get("mape", float("nan")),
                "overall_best_corr": overall_best.get("overall", {}).get("corr", float("nan")),
            }

            for channel in CHANNELS:
                channel_metrics = selected[channel]
                row[f"{channel}_mae"] = channel_metrics["mae"]
                row[f"{channel}_rmse"] = channel_metrics["rmse"]
                row[f"{channel}_mape"] = channel_metrics["mape"]
                row[f"{channel}_corr"] = channel_metrics["corr"]

            rows.append(row)

    full_by_horizon = {
        row["horizon"]: row for row in rows if row["experiment_tag"] == "core_full"
    }
    for row in rows:
        full_row = full_by_horizon[row["horizon"]]
        row["delta_mae_vs_full"] = row["selected_overall_mae"] - full_row["selected_overall_mae"]
        row["delta_rmse_vs_full"] = row["selected_overall_rmse"] - full_row["selected_overall_rmse"]
        row["delta_mape_vs_full"] = row["selected_overall_mape"] - full_row["selected_overall_mape"]
        for channel in CHANNELS:
            row[f"{channel}_delta_mae_vs_full"] = row[f"{channel}_mae"] - full_row[f"{channel}_mae"]

    return rows


def write_overall_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "horizon",
                "experiment_tag",
                "description",
                "output_dir",
                "final_export_source",
                "selected_prefix",
                "selected_overall_mae",
                "delta_mae_vs_full",
                "selected_overall_rmse",
                "delta_rmse_vs_full",
                "selected_overall_mape",
                "delta_mape_vs_full",
                "selected_overall_corr",
                "overall_best_mae",
                "overall_best_rmse",
                "overall_best_mape",
                "overall_best_corr",
                "electricity_mae",
                "electricity_delta_mae_vs_full",
                "electricity_rmse",
                "electricity_mape",
                "electricity_corr",
                "cooling_mae",
                "cooling_delta_mae_vs_full",
                "cooling_rmse",
                "cooling_mape",
                "cooling_corr",
                "heating_mae",
                "heating_delta_mae_vs_full",
                "heating_rmse",
                "heating_mape",
                "heating_corr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_channel_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "horizon",
                "experiment_tag",
                "description",
                "channel",
                "final_export_source",
                "mae",
                "delta_mae_vs_full",
                "rmse",
                "mape",
                "corr",
            ],
        )
        writer.writeheader()
        for row in rows:
            for channel in CHANNELS:
                writer.writerow(
                    {
                        "horizon": row["horizon"],
                        "experiment_tag": row["experiment_tag"],
                        "description": row["description"],
                        "channel": channel,
                        "final_export_source": row["final_export_source"],
                        "mae": row[f"{channel}_mae"],
                        "delta_mae_vs_full": row[f"{channel}_delta_mae_vs_full"],
                        "rmse": row[f"{channel}_rmse"],
                        "mape": row[f"{channel}_mape"],
                        "corr": row[f"{channel}_corr"],
                    }
                )


def write_mae_matrix_csv(rows: list[dict[str, object]], path: Path) -> None:
    by_experiment: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_experiment[row["experiment_tag"]][row["horizon"]] = row

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_tag",
                "description",
                "mae_h24",
                "delta_h24",
                "mae_h48",
                "delta_h48",
                "mae_h72",
                "delta_h72",
                "mae_h96",
                "delta_h96",
            ],
        )
        writer.writeheader()
        for tag, desc in EXPERIMENTS:
            item = {
                "experiment_tag": tag,
                "description": desc,
            }
            for horizon in HORIZONS:
                row = by_experiment[tag][horizon]
                item[f"mae_h{horizon}"] = row["selected_overall_mae"]
                item[f"delta_h{horizon}"] = row["delta_mae_vs_full"]
            writer.writerow(item)


def write_single_horizon_markdown(rows: list[dict[str, object]], horizon: int, path: Path) -> None:
    subset = [row for row in rows if row["horizon"] == horizon]
    subset.sort(key=lambda item: EXPERIMENTS.index((item["experiment_tag"], item["description"])))

    full_row = next(row for row in subset if row["experiment_tag"] == "core_full")
    ablations = [row for row in subset if row["experiment_tag"] != "core_full"]
    worst_mae = max(ablations, key=lambda item: item["delta_mae_vs_full"])
    best_mae = min(ablations, key=lambda item: item["delta_mae_vs_full"])

    with path.open("w", encoding="utf-8") as f:
        f.write("# run_ablation_core_new.ps1 结果整理\n\n")
        f.write("- 脚本: `run_ablation_core_new.ps1`\n")
        f.write("- 模型: `KAN_TQNet`\n")
        f.write(f"- 预测步长: `{horizon}`\n")
        f.write("- 随机种子: `0`\n")
        f.write("- 训练轮数: `60`\n")
        f.write("- 设备: `cuda:0`\n\n")

        f.write("## 总体结果\n\n")
        f.write("注: `Δ` 为相对 `core_full` 的变化, 指标越小越好, 因此正值表示变差。\n\n")
        f.write("| 实验 | 说明 | 最终来源 | MAE | ΔMAE | RMSE | ΔRMSE | MAPE | ΔMAPE | Corr |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in subset:
            f.write(
                "| {experiment_tag} | {description} | {final_export_source} | {mae} | {dmae} | {rmse} | {drmse} | {mape} | {dmape} | {corr} |\n".format(
                    experiment_tag=row["experiment_tag"],
                    description=row["description"],
                    final_export_source=row["final_export_source"],
                    mae=fmt(row["selected_overall_mae"]),
                    dmae=fmt_delta(row["delta_mae_vs_full"]),
                    rmse=fmt(row["selected_overall_rmse"]),
                    drmse=fmt_delta(row["delta_rmse_vs_full"]),
                    mape=fmt(row["selected_overall_mape"]),
                    dmape=fmt_delta(row["delta_mape_vs_full"]),
                    corr=fmt(row["selected_overall_corr"]),
                )
            )

        f.write("\n## 分通道 MAE\n\n")
        f.write("| 实验 | Electricity | Δ | Cooling | Δ | Heating | Δ |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in subset:
            f.write(
                "| {experiment_tag} | {e_mae} | {e_delta} | {c_mae} | {c_delta} | {h_mae} | {h_delta} |\n".format(
                    experiment_tag=row["experiment_tag"],
                    e_mae=fmt(row["electricity_mae"]),
                    e_delta=fmt_delta(row["electricity_delta_mae_vs_full"]),
                    c_mae=fmt(row["cooling_mae"]),
                    c_delta=fmt_delta(row["cooling_delta_mae_vs_full"]),
                    h_mae=fmt(row["heating_mae"]),
                    h_delta=fmt_delta(row["heating_delta_mae_vs_full"]),
                )
            )

        f.write("\n## 简要结论\n\n")
        f.write(f"- `core_full` 最优, 总体 MAE 为 `{fmt(full_row['selected_overall_mae'])}`。\n")
        f.write(
            f"- 影响最大的单项消融是 `{worst_mae['experiment_tag']}`, 相比完整模型 MAE 变化 `{fmt_delta(worst_mae['delta_mae_vs_full'])}`。\n"
        )
        f.write(
            f"- 影响最小的单项消融是 `{best_mae['experiment_tag']}`, 相比完整模型 MAE 变化 `{fmt_delta(best_mae['delta_mae_vs_full'])}`。\n"
        )
        f.write("- 表中 `final export source` 保留了最终导出结果来源。\n")


def write_combined_markdown(rows: list[dict[str, object]], path: Path) -> None:
    by_horizon: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_experiment: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_horizon[row["horizon"]].append(row)
        by_experiment[row["experiment_tag"]][row["horizon"]] = row

    with path.open("w", encoding="utf-8") as f:
        f.write("# run_ablation_core_new.ps1 合并结果整理\n\n")
        f.write("- 脚本: `run_ablation_core_new.ps1`\n")
        f.write("- 模型: `KAN_TQNet`\n")
        f.write("- 预测步长: `24 / 48 / 72 / 96`\n")
        f.write("- 随机种子: `0`\n")
        f.write("- 训练轮数: `60`\n")
        f.write("- 设备: `cuda:0`\n\n")

        f.write("## MAE 总表\n\n")
        f.write("注: 单元格格式为 `MAE (Δ)`，其中 `Δ` 为相对同步长 `core_full` 的变化。\n\n")
        f.write("| 实验 | 说明 | 24 | 48 | 72 | 96 |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for tag, desc in EXPERIMENTS:
            values = []
            for horizon in HORIZONS:
                row = by_experiment[tag][horizon]
                values.append(f"{fmt(row['selected_overall_mae'])} ({fmt_delta(row['delta_mae_vs_full'])})")
            f.write(f"| {tag} | {desc} | {' | '.join(values)} |\n")

        f.write("\n## 各步长最佳与最差消融\n\n")
        f.write("| 步长 | 完整模型 MAE | 影响最大消融 | ΔMAE | 影响最小消融 | ΔMAE |\n")
        f.write("| --- | ---: | --- | ---: | --- | ---: |\n")
        for horizon in HORIZONS:
            subset = by_horizon[horizon]
            full_row = next(row for row in subset if row["experiment_tag"] == "core_full")
            ablations = [row for row in subset if row["experiment_tag"] != "core_full"]
            worst_mae = max(ablations, key=lambda item: item["delta_mae_vs_full"])
            best_mae = min(ablations, key=lambda item: item["delta_mae_vs_full"])
            f.write(
                f"| {horizon} | {fmt(full_row['selected_overall_mae'])} | {worst_mae['experiment_tag']} | {fmt_delta(worst_mae['delta_mae_vs_full'])} | {best_mae['experiment_tag']} | {fmt_delta(best_mae['delta_mae_vs_full'])} |\n"
            )

        f.write("\n## 观察\n\n")
        for horizon in HORIZONS:
            subset = by_horizon[horizon]
            full_row = next(row for row in subset if row["experiment_tag"] == "core_full")
            ablations = [row for row in subset if row["experiment_tag"] != "core_full"]
            worst_mae = max(ablations, key=lambda item: item["delta_mae_vs_full"])
            best_mae = min(ablations, key=lambda item: item["delta_mae_vs_full"])
            f.write(
                f"- 步长 `{horizon}`: `core_full` MAE 为 `{fmt(full_row['selected_overall_mae'])}`, 最伤性能的是 `{worst_mae['experiment_tag']}` ({fmt_delta(worst_mae['delta_mae_vs_full'])}), 影响最小的是 `{best_mae['experiment_tag']}` ({fmt_delta(best_mae['delta_mae_vs_full'])})。\n"
            )

        f.write("- 当前 28 组实验的 `final export source` 均来自日志末尾记录，若同一日志存在重复 final 区段，脚本按最后一次记录取值。\n")


def main() -> None:
    rows = collect_rows()
    rows.sort(key=lambda item: (item["horizon"], EXPERIMENTS.index((item["experiment_tag"], item["description"]))))

    write_overall_csv(rows, PROJECT_ROOT / "core_ablation_new_all_overall.csv")
    write_channel_csv(rows, PROJECT_ROOT / "core_ablation_new_all_channels.csv")
    write_mae_matrix_csv(rows, PROJECT_ROOT / "core_ablation_new_mae_matrix.csv")
    write_combined_markdown(rows, PROJECT_ROOT / "core_ablation_new_all_summary.md")

    write_single_horizon_markdown(
        [row for row in rows if row["horizon"] == 96],
        96,
        PROJECT_ROOT / "core_ablation_new_summary.md",
    )
    write_overall_csv(
        [row for row in rows if row["horizon"] == 96],
        PROJECT_ROOT / "core_ablation_new_overall.csv",
    )
    write_channel_csv(
        [row for row in rows if row["horizon"] == 96],
        PROJECT_ROOT / "core_ablation_new_channels.csv",
    )


if __name__ == "__main__":
    main()
