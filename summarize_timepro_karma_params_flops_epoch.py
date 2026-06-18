from __future__ import annotations

import csv
import re
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from get_config import get_config
from test_model.TimePro import Model as TimePro
from test_model.KARMA import Model as KARMA


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
MODELS = {
    "TimePro": TimePro,
    "KARMA": KARMA,
}
HORIZONS = [24, 48, 72, 96]


def extract_epoch_time(result_dir: Path) -> float:
    text = (result_dir / "data.txt").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\| end of epoch\s+1 \| time:\s*([0-9.]+)s", text)
    if not m:
        raise RuntimeError(f"epoch time not found in {result_dir / 'data.txt'}")
    return float(m.group(1))


def compute_stats(model_name: str, horizon: int) -> dict[str, float]:
    cfg = get_config(model_name)
    cfg.pred_len = horizon
    model_cls = MODELS[model_name]
    model = model_cls(cfg).eval()

    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 1, 12, cfg.seq_len)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False) as prof:
        with torch.no_grad():
            _ = model(x)
    flops = sum((getattr(evt, "flops", 0) or 0) for evt in prof.key_averages())

    result_dir = OUTPUT_ROOT / f"{model_name}_{horizon}_168_quick_{model_name.lower()}_h{horizon}_e1" / "result"
    epoch_time = extract_epoch_time(result_dir)

    return {
        "Model": model_name,
        "Horizon": horizon,
        "Params": params,
        "Params_M": params / 1e6,
        "FLOPs": flops,
        "FLOPs_G": flops / 1e9,
        "EpochTime_s": epoch_time,
    }


def main() -> None:
    rows = []
    for model_name in MODELS:
        for horizon in HORIZONS:
            rows.append(compute_stats(model_name, horizon))

    rows.sort(key=lambda r: (r["Model"], r["Horizon"]))

    out_csv = OUTPUT_ROOT / "timepro_karma_params_flops_epoch_summary.csv"
    out_md = OUTPUT_ROOT / "timepro_karma_params_flops_epoch_summary.md"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Model", "Horizon", "Params", "Params_M", "FLOPs", "FLOPs_G", "EpochTime_s"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Model | Horizon | Params (M) | FLOPs (G) | Epoch time (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Model']} | {row['Horizon']} | {row['Params_M']:.3f} "
            f"| {row['FLOPs_G']:.3f} | {row['EpochTime_s']:.2f} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_csv)
    print(out_md)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
