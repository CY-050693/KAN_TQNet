import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run one forecasting model for multiple horizons.')
    parser.add_argument('--model_name', type=str, default='KAN_TQNet')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--horizons', type=int, nargs='+', default=[24, 48, 72, 96])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_script = root / 'train.py'

    for horizon in args.horizons:
        cmd = [
            args.python,
            str(train_script),
            '--model_name', args.model_name,
            '--device', args.device,
            '--epochs', str(args.epochs),
            '--pred_len_override', str(horizon),
        ]
        print(f'=== Start horizon {horizon} ===', flush=True)
        subprocess.run(cmd, cwd=root, check=True)
        print(f'=== Finished horizon {horizon} ===', flush=True)


if __name__ == '__main__':
    main()
