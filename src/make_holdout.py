"""Prepare a chronological split from B04 labeled training observations."""

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate import TRUTH_A, TRUTH_B
from inputs import find_inputs


def split_file(source, destination, name, validation_start, validation_end, train_end):
    truth_fields = TRUTH_A if name == "A" else TRUTH_B
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not set(truth_fields + ("datetime",)) <= set(fields):
            raise ValueError(f"Missing or duplicate labeled input fields: {source}")
        hidden = {"target_id", "target_class", "target_state"}
        test_fields = [field for field in fields if field not in hidden]
        with (
            (destination / "train" / f"{name}.csv").open("w", encoding="utf-8", newline="") as fit_file,
            (destination / "input" / f"{name}.csv").open("w", encoding="utf-8", newline="") as test_file,
            (destination / "truth" / f"{name}.csv").open("w", encoding="utf-8", newline="") as truth_file,
        ):
            fit = csv.DictWriter(fit_file, fields)
            test = csv.DictWriter(test_file, test_fields, extrasaction="ignore")
            truth = csv.DictWriter(truth_file, truth_fields, extrasaction="ignore")
            for writer in (fit, test, truth):
                writer.writeheader()
            counts = {"train": 0, "validation": 0, "excluded": 0}
            for row in reader:
                if None in row or None in row.values() or row["data_version"] != "train":
                    raise ValueError(f"{source}:{reader.line_num}: invalid labeled row")
                time = datetime.fromisoformat(row["datetime"])
                if time < train_end:
                    fit.writerow(row)
                    counts["train"] += 1
                elif validation_start <= time < validation_end:
                    row["data_version"] = "test"
                    test.writerow(row)
                    truth.writerow(row)
                    counts["validation"] += 1
                else:
                    counts["excluded"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--validation-start", required=True, type=datetime.fromisoformat)
    parser.add_argument("--validation-end", required=True, type=datetime.fromisoformat)
    parser.add_argument("--gap-seconds", default=10, type=int)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir must be a new directory")
    if args.gap_seconds < 0 or args.validation_start >= args.validation_end:
        parser.error("expected a nonnegative gap and a validation end after the start")
    train_end = args.validation_start - timedelta(seconds=args.gap_seconds)
    try:
        paths = find_inputs(args.data_root, "train")
        args.out_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".b04-holdout-", dir=args.out_dir.parent) as scratch:
            staging = Path(scratch) / "split"
            for directory in ("train", "input", "truth"):
                (staging / directory).mkdir(parents=True)
            counts = {
                sensor: split_file(path, staging, sensor, args.validation_start, args.validation_end, train_end)
                for sensor, path in paths.items()
            }
            for sensor, count in counts.items():
                if not count["train"] or not count["validation"]:
                    raise ValueError(f"{sensor}: train and validation must both contain observations")
            report = {
                "train_end_exclusive": train_end.isoformat(),
                "validation_start_inclusive": args.validation_start.isoformat(),
                "validation_end_exclusive": args.validation_end.isoformat(),
                "source_basenames": {sensor: path.name for sensor, path in paths.items()},
                "counts": counts,
            }
            (staging / "split.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            staging.rename(args.out_dir)
        print(args.out_dir / "split.json")
    except (OSError, TypeError, ValueError) as error:
        parser.exit(2, f"Holdout creation failed: {error}\n")


if __name__ == "__main__":
    main()
