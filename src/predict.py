"""Run the B04 candidate and write one pair of competition CSVs per scene."""

import csv
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from evaluate import CLASSES, LINKS, PREDICTIONS, PULSE_KEY, STATES
from geometry import TrackMatcher, load_tracks
from gpu_check import cuda_device
from temporal_model import CONTEXT_WIDTH_US, is_ordered, load_model, pulse_batches, pulse_features


SCENE_KEY = list(PULSE_KEY[:2])


def batches_with_context(batches, width):
    """Include neighboring blocks so peers stay visible at CSV batch boundaries."""
    if not width:
        for batch in batches:
            yield batch, 0, len(batch)
        return
    previous = pd.DataFrame()
    current = next(batches, None)
    if current is None:
        return
    for following in chain(batches, (None,)):
        start = (current.toa.iloc[0] // width - 1) * width
        stop = (current.toa.iloc[-1] // width + 2) * width
        left = previous[previous.toa >= start] if len(previous) else previous
        right = following[following.toa < stop] if following is not None else None
        batch = pd.concat((left, current, right), ignore_index=True)
        yield batch, len(left), len(left) + len(current)
        previous = current
        current = following


@torch.inference_mode()
def classified_batches(path, model, device, limit, ordered, *, use_arrivals=False):
    batches = pulse_batches(path, limit=limit, ordered=ordered, feature_version=model.feature_version)
    width = CONTEXT_WIDTH_US[model.feature_version] if use_arrivals else 0
    for batch, start, stop in batches_with_context(batches, width):
        values = torch.as_tensor(pulse_features(batch, model.feature_version), device=device)
        for key, rows in batch.groupby(SCENE_KEY, sort=False):
            row_indices = rows.index.to_numpy(copy=True)
            core = (row_indices >= start) & (row_indices < stop)
            if not core.any():
                continue
            index = torch.as_tensor(row_indices, device=device)
            types, states = model(values[index])
            yield key, rows.reset_index(drop=True), types.softmax(1), states.argmax(1).cpu().numpy(), core


class SceneOutput:
    def __init__(self, directory, key):
        self.key = key
        if any(not item or Path(item).name != item for item in key):
            raise ValueError(f"invalid scene or data version: {key}")
        prefix = "_".join(key)
        self.pulses = directory / f"{prefix}_pulse_predictions.csv"
        self.links = directory / f"{prefix}_object_track_links.csv"
        self.objects = {}

    def append(self, batch, types, states, indices, tracks):
        class_labels = np.asarray(CLASSES)[types.argmax(1).cpu().numpy()]
        group = np.char.add("object-u-", class_labels).astype(object)
        for index in np.unique(indices[indices >= 0]):
            track = tracks[index].identifier
            identifier = f"object-b-{index + 1}"
            group[indices == index] = identifier
            self.objects[identifier] = track
        self.objects.update((identifier, "") for identifier in np.unique(group[indices < 0]))
        result = batch[list(PULSE_KEY)].assign(
            local_cluster_id=group, object_id=group,
            target_class=class_labels, target_state=np.asarray(STATES)[states],
        )
        result.to_csv(self.pulses, columns=PREDICTIONS, mode="a", index=False, header=not self.pulses.exists())

    def finish(self):
        with self.links.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(LINKS)
            writer.writerows((*self.key, obj, track, "matched" if track else "no_match")
                             for obj, track in sorted(self.objects.items()))


def predict(path_a, path_b, checkpoint, destination, limit=None, *, association="bearing"):
    device = cuda_device()
    model = load_model(checkpoint, device)
    ordered = is_ordered(path_a, limit)
    use_arrivals = association == "arrival"
    matchers = {
        key: TrackMatcher(tracks, device, use_arrivals=use_arrivals)
        for key, tracks in load_tracks(path_b).items()
    }
    unmatched = TrackMatcher([], device)
    with torch.inference_mode():
        for key, rows, types, _, core in classified_batches(
            path_a, model, device, limit, ordered, use_arrivals=use_arrivals
        ):
            mask = torch.as_tensor(core, device=device) if use_arrivals else None
            matchers.get(key, unmatched).observe(rows, types, mask)
        for matcher in matchers.values():
            matcher.finish()

        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".b04-prediction-", dir=destination) as temporary:
            directory = Path(temporary)
            outputs = {}
            count = 0
            for key, rows, types, states, core in classified_batches(
                path_a, model, device, limit, ordered, use_arrivals=use_arrivals
            ):
                if key not in outputs:
                    outputs[key] = SceneOutput(directory, key)
                matcher = matchers.get(key, unmatched)
                matches = matcher.match(rows, types)
                if use_arrivals:
                    mask = torch.as_tensor(core, device=device)
                    rows, types, states, matches = rows.iloc[core], types[mask], states[core], matches[core]
                outputs[key].append(rows, types, states, matches, matcher.tracks)
                count += len(rows)
            if not count:
                raise ValueError("empty A sensor input")
            for output in outputs.values():
                output.finish()
            for file in sorted(directory.iterdir()):
                file.replace(destination / file.name)
    print(f"{destination} ({count} pulses across {len(outputs)} scenes)")
