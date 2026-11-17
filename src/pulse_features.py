"""Build GPU-derived pulse features in a bounded stream, retaining pulse identity."""

import csv
import math
from itertools import batched, islice
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gpu_check import cuda_device
from inputs import normalize_columns


TEXT_FIELDS = (
    "scenario_id", "data_version", "pulse_id", "platform_id", "datetime",
    "target_id", "target_class", "target_state",
)
MEASUREMENTS = (
    "toa", "param1", "param2", "param3", "doa",
    "sensor_longitude", "sensor_latitude", "sensor_height",
)
SCHEMA = pa.schema(
    [(field, pa.string()) for field in TEXT_FIELDS]
    + [(field, pa.float64()) for field in MEASUREMENTS]
    + [("doa_sin", pa.float32()), ("doa_cos", pa.float32())]
)


def build_batch(rows: tuple[dict[str, str], ...], device: torch.device) -> pa.Table:
    columns: dict[str, object] = {}
    for field in TEXT_FIELDS:
        columns[field] = [(row.get(field) or "").strip() or None for row in rows]
    for field in MEASUREMENTS:
        values = [float(row[field]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{field} has a non-finite value")
        columns[field] = values

    angles = torch.deg2rad(torch.tensor(columns["doa"], device=device, dtype=torch.float32))
    columns["doa_sin"] = angles.sin().cpu().numpy()
    columns["doa_cos"] = angles.cos().cpu().numpy()
    return pa.Table.from_pydict(columns, schema=SCHEMA)


def write_pulse_features(source: Path, destination: Path, limit: int | None = None) -> int:
    device = cuda_device()
    rows_written = 0
    with NamedTemporaryFile(
        dir=destination.parent, prefix=".pulse-features-", suffix=".parquet", delete=False
    ) as pending:
        pending_path = Path(pending.name)

    try:
        with (
            source.open(encoding="utf-8-sig", newline="") as stream,
            pq.ParquetWriter(pending_path, SCHEMA, compression="zstd") as writer,
        ):
            reader = csv.DictReader(stream)
            reader.fieldnames = normalize_columns(reader.fieldnames)
            for rows in batched(islice(reader, limit), 65_536):
                if any(None in row or None in row.values() for row in rows):
                    raise ValueError(f"{source}:{reader.line_num} has an incomplete CSV batch")
                writer.write_table(build_batch(rows, device))
                rows_written += len(rows)

        if not rows_written:
            raise ValueError(f"empty pulse table: {source}")
        pending_path.replace(destination)
        return rows_written
    finally:
        pending_path.unlink(missing_ok=True)
