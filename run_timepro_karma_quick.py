import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Quick multi-horizon run for TimePro and KARMA.")
    parser.add_argument("--models", nargs="+", default=["TimePro", "KARMA"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[24, 48, 72, 96])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--exp_tag", type=str, default="quick")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    train_script = root / "train.py"
    failures = []

    for model_name in args.models:
        for horizon in args.horizons:
            tag = f"{args.exp_tag}_{model_name.lower()}_h{horizon}_e{args.epochs}"
            cmd = [
                args.python,
                str(train_script),
                "--model_name",
                model_name,
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--pred_len_override",
                str(horizon),
                "--exp_tag",
                tag,
            ]
            print(f"\n=== Start {model_name} horizon={horizon} epochs={args.epochs} ===", flush=True)
            print(" ".join(cmd), flush=True)
            try:
                subprocess.run(cmd, cwd=root, check=True)
            except subprocess.CalledProcessError as exc:
                failures.append((model_name, horizon, exc.returncode))
                print(f"!!! Failed {model_name} horizon={horizon}: returncode={exc.returncode}", flush=True)
                if not args.continue_on_error:
                    raise
            else:
                print(f"=== Finished {model_name} horizon={horizon} ===", flush=True)

    if failures:
        print("\nFailures:")
        for model_name, horizon, code in failures:
            print(f"- {model_name} horizon={horizon}: returncode={code}")
        sys.exit(1)

    print("\nAll quick multi-horizon runs finished.")


if __name__ == "__main__":
    main()
