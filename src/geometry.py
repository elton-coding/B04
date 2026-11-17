"""Associate pulse observations with temporally supported B-sensor bearings."""

import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from evaluate import CLASSES
from inputs import normalize_columns
from temporal_model import context_groups, read_csv


TRACK_FIELDS = ("scenario_id", "data_version", "track_id", "datetime",
                "target_longitude", "target_latitude")
MAX_GAP_SECONDS = 10
EDGE_SECONDS = 3


def seconds(values):
    times = pd.to_datetime(values, format="ISO8601").to_numpy(dtype="datetime64[ms]")
    if np.isnat(times).any():
        raise ValueError("missing observation time")
    return times.astype(np.int64) / 1000


@dataclass
class Track:
    identifier: str
    time: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    label: str


def load_tracks(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        names = normalize_columns(next(csv.reader(stream), []))
    optional = ("target_class",) if "target_class" in names else ()
    tracks = read_csv(path, TRACK_FIELDS + optional, TRACK_FIELDS[:4] + optional)
    tracks["time"] = seconds(tracks.datetime)
    position = ["target_longitude", "target_latitude"]
    tracks[position] = tracks[position].astype("float64")
    if not np.isfinite(tracks[position].to_numpy()).all():
        raise ValueError(f"{path}: track coordinates must be finite")
    catalog = {}
    for (scene, version, identifier), rows in tracks.groupby(list(TRACK_FIELDS[:3]), sort=False):
        if not scene or not version or not identifier:
            raise ValueError(f"{path}: empty track identity")
        locations = rows.groupby("time")[position].mean()
        labels = set(rows.target_class.str.strip()) & set(CLASSES) if optional else set()
        label = next(iter(labels)) if len(labels) == 1 else ""
        catalog.setdefault((scene, version), []).append(Track(
            identifier, locations.index.to_numpy(), locations.target_longitude.to_numpy(),
            locations.target_latitude.to_numpy(), label,
        ))
    return catalog


def interpolate(track, time):
    right = np.searchsorted(track.time, time).clip(0, len(track.time) - 1)
    left = np.maximum(right - 1, 0)
    next_gap = np.abs(time - track.time[right])
    prev_gap = np.abs(time - track.time[left])
    bridge = ((time >= track.time[0]) & (time <= track.time[-1])
              & (track.time[right] - track.time[left] <= MAX_GAP_SECONDS))
    covered = bridge | (np.minimum(next_gap, prev_gap) <= EDGE_SECONDS)
    nearest = np.where(next_gap <= prev_gap, right, left)
    return (
        np.where(bridge, np.interp(time, track.time, track.longitude), track.longitude[nearest]),
        np.where(bridge, np.interp(time, track.time, track.latitude), track.latitude[nearest]),
        covered,
    )


def averaged_doa(batch, types):
    rad = np.deg2rad(batch.doa.to_numpy())
    groups = context_groups(batch, 32_000, 8) + [types]
    vectors = pd.DataFrame({"sin": np.sin(rad), "cos": np.cos(rad)})
    mean = vectors.groupby(groups, sort=False).transform("mean")
    return np.rad2deg(np.arctan2(mean.sin.to_numpy(), mean.cos.to_numpy()))


def wrapped_error(first, second):
    return ((first - second + 180) % 360 - 180).abs()


def bearing_errors(batch, tracks, types, device):
    time, ix = np.unique(seconds(batch.datetime), return_inverse=True)
    longitude, latitude, supported = map(np.column_stack, zip(*(interpolate(track, time) for track in tracks)))
    ix = torch.as_tensor(ix, device=device)
    lng = torch.as_tensor(longitude, device=device)[ix]
    lat = torch.as_tensor(latitude, device=device)[ix]
    coverage = torch.as_tensor(supported, device=device)[ix]
    source = torch.as_tensor(
        batch[["sensor_longitude", "sensor_latitude"]].to_numpy(copy=True), device=device
    )
    delta = torch.deg2rad(lng - source[:, :1])
    lat = torch.deg2rad(lat)
    origin = torch.deg2rad(source[:, 1:])
    bearings = torch.rad2deg(torch.atan2(
        delta.sin() * lat.cos(),
        origin.cos() * lat.sin() - origin.sin() * lat.cos() * delta.cos(),
    )).float()
    raw = torch.as_tensor(batch.doa.to_numpy(copy=True), dtype=torch.float32, device=device)[:, None]
    smooth = torch.as_tensor(averaged_doa(batch, types), dtype=torch.float32, device=device)[:, None]
    error = wrapped_error(bearings, smooth).masked_fill(~coverage, torch.inf)
    return error, wrapped_error(bearings, raw)


class TrackMatcher:
    def __init__(self, tracks, device):
        self.tracks = tracks
        self.device = device
        self.evidence = torch.zeros((len(tracks), len(CLASSES)), device=device)

    def observe(self, batch, probabilities):
        if not self.tracks:
            return
        types = probabilities.argmax(1).cpu().numpy()
        error, _ = bearing_errors(batch, self.tracks, types, self.device)
        best, indices = error.min(1)
        if len(self.tracks) > 1:
            second = error.topk(2, largest=False).values[:, 1]
            selected = (best < 2.5) & (second > best + 4.0)
        else:
            selected = best < 2.5
        self.evidence.index_add_(0, indices[selected], probabilities[selected])

    def finish(self):
        count = self.evidence.sum(1, keepdim=True)
        self.prior = (self.evidence + 4) / (count + 4 * len(CLASSES))
        self.prior = 0.94 * self.prior + 0.06 / len(CLASSES)
        for index, track in enumerate(self.tracks):
            if track.label:
                self.prior[index].fill_(0.06 / (len(CLASSES) - 1))
                self.prior[index, CLASSES.index(track.label)] = 0.94

    def match(self, batch, probabilities):
        if not self.tracks:
            return np.full(len(batch), -1, dtype=np.int64)
        types = probabilities.argmax(1).cpu().numpy()
        error, raw = bearing_errors(batch, self.tracks, types, self.device)
        likelihood = probabilities @ self.prior.T
        # Uniform type evidence contributes zero cost. Bearing gates reject false matches.
        cost = (error / 2.5).square() + 0.18 * (raw / 2.5).square()
        cost -= (likelihood * len(CLASSES)).clamp_min(1e-5).log()
        cost.masked_fill_((error > 3.5) | (raw > 7), torch.inf)
        best, indices = cost.min(1)
        return indices.masked_fill(best > 2, -1).cpu().numpy()
