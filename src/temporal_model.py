"""Train and apply a CUDA classifier with time and bearing context."""

import argparse
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluate import CLASSES, PULSE_KEY, STATES
from gpu_check import cuda_device
from inputs import normalize_columns


SIGNAL = ("param1", "param2", "param3")
POSITION = ("sensor_longitude", "sensor_latitude", "sensor_height")
MEASUREMENTS = SIGNAL + ("toa", "doa") + POSITION
TEXT = PULSE_KEY + ("platform_id", "datetime")
LABELS = ("target_class", "target_state")
SEED = 42
FEATURE_VERSION = 1
CONTEXT_WIDTH_US = {1: 256_000, 2: 2_048_000}
CHUNK_SIZE = 262_144
STATE_KEEP_RATES = (0.12, 1.0, 0.30)
STATE_PRIOR_STRENGTH = 0.8


def read_csv(path, fields, strings, **options):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        names = normalize_columns(next(csv.reader(stream), []))
    if len(names) != len(set(names)) or set(fields) - set(names):
        raise ValueError(f"{path}: missing or duplicate fields")
    return pd.read_csv(
        path, header=0, names=names, usecols=lambda field: field in fields,
        dtype={field: "string" for field in strings}, keep_default_na=False, **options,
    )


def is_ordered(path, limit=None):
    previous = -np.inf
    for chunk in read_csv(path, ("toa",), (), chunksize=CHUNK_SIZE, nrows=limit):
        if chunk.empty:
            continue
        times = chunk.toa.astype("float64")
        if not np.isfinite(times.to_numpy()).all():
            raise ValueError(f"{path}: non-finite pulse arrival time")
        if times.iloc[0] < previous or not times.is_monotonic_increasing:
            return False
        previous = times.iloc[-1]
    return True


def pulse_batches(path, *, labeled=False, limit=None, ordered=None, feature_version=FEATURE_VERSION):
    fields = TEXT + MEASUREMENTS + (LABELS if labeled else ())
    if ordered is None:
        ordered = is_ordered(path, limit)
    if ordered:
        reader = read_csv(path, fields, TEXT + LABELS, chunksize=CHUNK_SIZE, nrows=limit)
    else:
        # Published CSVs are chronological. For unordered inputs, sort the
        # required columns once before calculating TOA neighbors and windows.
        frame = read_csv(path, fields, TEXT + LABELS, nrows=limit)
        frame["toa"] = frame.toa.astype("float64")
        frame = frame.sort_values("toa", kind="stable", ignore_index=True)
        reader = (frame.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(frame), CHUNK_SIZE))
    pending = pd.DataFrame()
    for batch in reader:
        if batch.empty:
            continue
        batch[list(MEASUREMENTS)] = batch[list(MEASUREMENTS)].astype("float64")
        if not np.isfinite(batch[list(MEASUREMENTS)].to_numpy()).all():
            raise ValueError(f"{path}: pulse observations must be finite")
        if (batch[list(TEXT)] == "").any().any():
            raise ValueError(f"{path}: empty pulse identity, platform or timestamp")
        # Hold back the last context block so CSV chunk boundaries do not split it.
        batch = pd.concat((pending, batch), ignore_index=True)
        block = np.floor_divide(batch.toa.to_numpy(), CONTEXT_WIDTH_US[feature_version]).astype("int64")
        boundary = np.r_[True, block[1:] != block[:-1]]
        cut = np.flatnonzero(boundary)[-1]
        pending = batch.iloc[cut:].copy()
        if cut:
            yield batch.iloc[:cut].reset_index(drop=True)
    if len(pending):
        yield pending.reset_index(drop=True)


def context_groups(batch, width_us, bearing_width, offset=0):
    return [
        batch.scenario_id, batch.data_version, batch.platform_id,
        np.floor_divide(batch.toa.to_numpy(), width_us).astype("int64"),
        np.floor(batch.param1.to_numpy() / 1000 + 0.5).astype("int64"),
        np.floor((batch.doa.to_numpy() + offset) % 360 / bearing_width).astype("int64"),
    ]


def burst_features(batch):
    toa = batch.toa
    signal_bin = np.floor((batch.param3.to_numpy() + 2) / 4).astype("int64")
    subwindow = (toa // 16_000).astype("int64")
    parts = []
    for width, angle, offset in ((64_000, 8, 4), (512_000, 10, 5), (2_048_000, 12, 0)):
        groups = context_groups(batch, width, angle, offset) + [signal_bin]
        arrivals = toa.groupby(groups, sort=False)
        gap = (toa - arrivals.shift()).clip(0, width)
        next_gap = (arrivals.shift(-1) - toa).clip(0, width)
        log_gaps = np.log1p(gap)
        gap_stats = log_gaps.groupby(groups, sort=False)
        occupied = subwindow.groupby(groups, sort=False).transform("nunique")
        count = arrivals.transform("size")
        start = arrivals.transform("min")
        finish = arrivals.transform("max")
        widths = batch.param2.groupby(groups, sort=False)
        parts.extend((
            np.log1p(gap.fillna(width)),
            np.log1p(next_gap.fillna(width)),
            gap_stats.transform("mean").fillna(np.log1p(width)),
            gap_stats.transform("std").fillna(0),
            (gap > 5_000).where(gap.notna()).astype(float).groupby(groups, sort=False)
                .transform("mean").fillna(0),
            np.log(occupied),
            np.log(count / occupied),
            np.log1p(toa - start),
            np.log1p(finish - toa),
            widths.transform("mean"),
            widths.transform("std").fillna(0),
        ))
    return parts


def pulse_features(batch, feature_version=FEATURE_VERSION):
    values = batch[list(SIGNAL)].astype("float32")
    parts = [values.to_numpy()]
    for width, angle, offset in ((32_000, 8, 0), (256_000, 12, 6)):
        groups = context_groups(batch, width, angle, offset)
        stats = values.groupby(groups, sort=False)
        times = batch.toa.groupby(groups, sort=False)
        parts.extend((
            stats.transform("mean").to_numpy(),
            stats.transform("std").fillna(0).to_numpy(),
            np.log1p(stats.param1.transform("size").to_numpy())[:, None],
            # Missing neighbors are represented by the window width.
            np.log1p((batch.toa - times.shift()).clip(0, width).fillna(width).to_numpy())[:, None],
            np.log1p((times.shift(-1) - batch.toa).clip(0, width).fillna(width).to_numpy())[:, None],
        ))
    if feature_version == 2:
        parts.extend(burst_features(batch))
    return np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)


class PulseClassifier(nn.Module):
    def __init__(self, features, feature_version=FEATURE_VERSION):
        super().__init__()
        self.feature_version = feature_version
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer("state_logit_bias", torch.zeros(len(STATES)), persistent=False)
        self.encoder = nn.Sequential(
            nn.Linear(features, 96), nn.SiLU(),
            nn.Linear(96, 96), nn.SiLU(),
            nn.Linear(96, 64), nn.SiLU(),
        )
        self.type_head = nn.Linear(64, len(CLASSES))
        self.state_head = nn.Linear(64 + len(CLASSES), len(STATES))

    def forward(self, values):
        hidden = self.encoder(((values - self.center) / self.scale).clamp(-15, 15))
        types = self.type_head(hidden)
        states = self.state_head(torch.cat((hidden, types.detach().softmax(1)), 1))
        return types, states - self.state_logit_bias


def label_codes(batch, name, labels):
    codes = pd.Categorical(batch[name].str.strip(), categories=labels).codes
    if (codes < 0).any():
        raise ValueError(f"invalid or missing {name} in training data")
    return codes


def train(path, destination, *, epochs=16, limit=None, feature_version=FEATURE_VERSION):
    device = cuda_device()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    features, types, states = [], [], []
    type_counts = np.zeros(len(CLASSES), dtype=np.int64)
    state_counts = np.zeros(len(STATES), dtype=np.int64)
    for batch in pulse_batches(path, labeled=True, limit=limit, feature_version=feature_version):
        t = label_codes(batch, "target_class", CLASSES)
        s = label_codes(batch, "target_state", STATES)
        type_counts += np.bincount(t, minlength=len(CLASSES))
        state_counts += np.bincount(s, minlength=len(STATES))
        # Sample after constructing neighbors. Retain extra labeled tracking pulses.
        keep = rng.random(len(batch)) < np.asarray(STATE_KEEP_RATES)[s]
        features.append(pulse_features(batch, feature_version)[keep])
        types.append(t[keep])
        states.append(s[keep])
    if not features or not any(len(part) for part in features):
        raise ValueError("empty labeled training input")
    x = torch.as_tensor(np.concatenate(features), device=device)
    y = torch.as_tensor(np.concatenate(types), device=device, dtype=torch.long)
    z = torch.as_tensor(np.concatenate(states), device=device, dtype=torch.long)
    model = PulseClassifier(x.shape[1], feature_version).to(device)
    model.center.copy_(x.mean(0))
    model.scale.copy_(x.std(0, unbiased=False).clamp_min(0.001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    type_freq = y.bincount(minlength=len(CLASSES)).float()
    state_freq = z.bincount(minlength=len(STATES)).float()
    type_loss = nn.CrossEntropyLoss(weight=(type_freq.max() / type_freq.clamp_min(1)).pow(0.45))
    state_loss = nn.CrossEntropyLoss(weight=(state_freq.max() / state_freq.clamp_min(1)).pow(0.5))
    for epoch in range(epochs):
        loss_sum = torch.zeros((), device=device)
        for ix in torch.randperm(len(x), device=device).split(8192):
            prediction, state = model(x[ix])
            loss = type_loss(prediction, y[ix]) + state_loss(state, z[ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(ix)
        print(f"epoch {epoch + 1}/{epochs} loss={loss_sum.item() / len(x):.5f}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save({
            "feature_version": feature_version, "model": model.state_dict(), "width": x.shape[1],
            "seed": SEED, "class_counts": type_counts.tolist(), "state_counts": state_counts.tolist(),
        }, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{destination} ({len(x)} sampled training pulses)")


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    feature_version = checkpoint["feature_version"]
    if feature_version not in CONTEXT_WIDTH_US:
        raise ValueError("checkpoint uses an unsupported pulse feature version")
    model = PulseClassifier(checkpoint["width"], feature_version).to(device)
    model.load_state_dict(checkpoint["model"])
    counts = torch.as_tensor(checkpoint["state_counts"], dtype=torch.float32, device=device)
    rates = torch.as_tensor(STATE_KEEP_RATES, device=device)
    # Retention multiplies a label's frequency by its keep rate; loss weights
    # then scale it by the inverse square root of its sampled frequency.
    log_weight = 0.5 * (rates.log() - counts.clamp_min(1).log())
    model.state_logit_bias.copy_(STATE_PRIOR_STRENGTH * (log_weight - log_weight[0]))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--feature-version", type=int, choices=tuple(CONTEXT_WIDTH_US), default=FEATURE_VERSION)
    args = parser.parse_args()
    if args.epochs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--epochs and --limit must be positive")
    try:
        train(args.train_a, args.output, epochs=args.epochs, limit=args.limit,
              feature_version=args.feature_version)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
