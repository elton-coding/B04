"""Discover, normalize, and audit the two B04 observation tables."""

import csv
from collections import Counter
from itertools import islice
from pathlib import Path


ALIASES = {
    "sensor_lontitude": "sensor_longitude",
    "target_lontitude": "target_longitude",
    "lontitude": "target_longitude",
    "longitude": "target_longitude",
    "latitude": "target_latitude",
    "height": "target_height",
}
COMMON = {"scenario_id", "data_version", "datetime"}
REQUIRED = {
    "A": COMMON | {
        "pulse_id", "platform_id", "toa", "param1", "param2", "param3", "doa",
        "sensor_longitude", "sensor_latitude", "sensor_height",
    },
    "B": COMMON | {
        "track_id", "target_type", "target_longitude", "target_latitude", "target_height",
    },
}


def normalize_columns(fields: list[str] | None) -> list[str]:
    return [ALIASES.get(field.strip(), field.strip()) for field in fields or ()]


def find_inputs(root: Path, split: str) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"input directory does not exist: {root}")

    candidates: dict[str, list[Path]] = {"A": [], "B": []}
    for path in sorted(root.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = normalize_columns(reader.fieldnames)
            for sensor, required in REQUIRED.items():
                if not required.issubset(columns):
                    continue
                if len(set(columns)) != len(columns):
                    raise ValueError(f"duplicate column after alias normalization: {path}")
                reader.fieldnames = columns
                first_row = next(reader, None)
                if first_row and first_row["data_version"] == split:
                    candidates[sensor].append(path)

    for sensor, paths in candidates.items():
        if len(paths) != 1:
            names = ", ".join(str(path) for path in paths) or "none"
            raise ValueError(f"expected one {split} {sensor} input under {root}, found {names}")
    return {sensor: paths[0] for sensor, paths in candidates.items()}


def summarize(path: Path, sensor: str, limit: int | None) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        reader.fieldnames = normalize_columns(reader.fieldnames)
        missing: Counter[str] = Counter()
        series: set[tuple[str, str]] = set()
        node_column = "platform_id" if sensor == "A" else "track_id"
        by_node: Counter[str] = Counter()
        rows = 0

        for row in islice(reader, limit):
            if None in row or None in row.values():
                raise ValueError(f"{path}:{reader.line_num} has the wrong number of columns")
            series.add((row["scenario_id"], row["data_version"]))
            by_node[row[node_column]] += 1
            missing.update(key for key, value in row.items() if not value.strip())
            rows += 1

    if rows == 0:
        raise ValueError(f"empty observation table: {path}")

    return {
        "file": path.name,
        "rows_read": rows,
        "columns": reader.fieldnames,
        "scenarios": sorted(series),
        "counts_by_" + node_column: dict(sorted(by_node.items())),
        "missing": dict(sorted(missing.items())),
    }


def audit_inputs(root: Path, split: str, limit: int | None = None) -> dict:
    paths = find_inputs(root, split)
    observations = {sensor: summarize(path, sensor, limit) for sensor, path in paths.items()}
    a_series = observations["A"]["scenarios"]
    b_series = observations["B"]["scenarios"]
    if any(not scenario.strip() or version != split for scenario, version in a_series + b_series):
        raise ValueError("A and B rows must have a scenario ID and the requested data version")
    if limit is None and a_series != b_series:
        raise ValueError("Full audit found different scenario IDs in the A and B files")
    return {"split": split, "row_limit_per_table": limit, "observations": observations}
