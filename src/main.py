"""B04 development entry point and input audit."""

import argparse
import json
import os
from pathlib import Path

from inputs import audit_inputs, find_inputs


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "audit", "features", "gpu-check"), default="all")
    parser.add_argument("--test", action="store_true", help="Select the test split")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("DATA_ROOT", "/input")))
    parser.add_argument("--out-dir", type=Path, default=Path(os.getenv("OUTPUT_DIR", "/output")))
    parser.add_argument("--limit", type=positive_int, help="Process at most this many rows per input")
    args = parser.parse_args()

    if args.task == "gpu-check":
        from gpu_check import check_cuda

        print(json.dumps(check_cuda(), indent=2))
        return

    if args.task == "all":
        parser.exit(2, "The prediction pipeline is under development. Use audit, features, or gpu-check.\n")

    split = "test" if args.test else "train"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.task == "features":
            from pulse_features import write_pulse_features

            source = find_inputs(args.data_root, split)["A"]
            output_path = args.out_dir / "pulse_features.parquet"
            rows = write_pulse_features(source, output_path, args.limit)
            print(f"{output_path} ({rows} pulses)")
            return
        report = audit_inputs(args.data_root, split, args.limit)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{args.task} failed: {error}\n")

    output_path = args.out_dir / "input_audit.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
