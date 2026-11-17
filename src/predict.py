"""Run the B04 candidate and write one pair of competition CSVs per scene."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from evaluate import CLASSES, LINKS, PREDICTIONS, PULSE_KEY, STATES
from geometry import TrackMatcher, load_tracks
from gpu_check import cuda_device
from temporal_model import is_ordered, load_model, pulse_batches, pulse_features


SCENE_KEY = list(PULSE_KEY[:2])


@torch.inference_mode()
def classified_batches(path, model, device, limit, ordered):
    for batch in pulse_batches(path, limit=limit, ordered=ordered):
        values = torch.as_tensor(pulse_features(batch), device=device)
        for key, rows in batch.groupby(SCENE_KEY, sort=False):
            index = torch.as_tensor(rows.index.to_numpy(copy=True), device=device)
            types, states = model(values[index])
            yield key, rows.reset_index(drop=True), types.softmax(1), states.argmax(1).cpu().numpy()


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


def predict(path_a, path_b, checkpoint, destination, limit=None):
    device = cuda_device()
    model = load_model(checkpoint, device)
    ordered = is_ordered(path_a, limit)
    matchers = {key: TrackMatcher(tracks, device) for key, tracks in load_tracks(path_b).items()}
    unmatched = TrackMatcher([], device)
    with torch.inference_mode():
        for key, rows, types, _ in classified_batches(path_a, model, device, limit, ordered):
            matchers.get(key, unmatched).observe(rows, types)
        for matcher in matchers.values():
            matcher.finish()

        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".b04-prediction-", dir=destination) as temporary:
            directory = Path(temporary)
            outputs = {}
            count = 0
            for key, rows, types, states in classified_batches(path_a, model, device, limit, ordered):
                if key not in outputs:
                    outputs[key] = SceneOutput(directory, key)
                matcher = matchers.get(key, unmatched)
                outputs[key].append(rows, types, states, matcher.match(rows, types), matcher.tracks)
                count += len(rows)
            if not count:
                raise ValueError("empty A sensor input")
            for output in outputs.values():
                output.finish()
            for file in sorted(directory.iterdir()):
                file.replace(destination / file.name)
    print(f"{destination} ({count} pulses across {len(outputs)} scenes)")
