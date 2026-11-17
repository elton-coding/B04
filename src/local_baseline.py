"""Generate two simple references for local validation on a chronological split."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from evaluate import CLASSES, LINKS, PREDICTIONS, PULSE_KEY, STATES
from gpu_check import cuda_device


FEATURES = ("param1", "param2", "param3")
CHUNK_SIZE = 131_072


class DiagonalGaussian:
    def __init__(self, labels, device):
        self.labels = labels
        self.device = device
        self.count = torch.zeros(len(labels), dtype=torch.float64, device=device)
        self.sum = torch.zeros((len(labels), len(FEATURES)), dtype=torch.float64, device=device)
        self.square_sum = torch.zeros_like(self.sum)

    def update(self, values, labels):
        codes = pd.Categorical(labels, categories=self.labels).codes
        if (codes < 0).any():
            raise ValueError("Invalid or missing training label")
        codes = torch.tensor(codes, dtype=torch.long, device=self.device)
        self.count.index_add_(0, codes, torch.ones_like(codes, dtype=torch.float64))
        self.sum.index_add_(0, codes, values)
        self.square_sum.index_add_(0, codes, values.square())

    def majority(self):
        return self.labels[self.count.argmax().item()]

    def predict(self, values):
        present = self.count > 0
        if not present.any().item():
            raise ValueError("No labeled observations were used to fit the model")
        total = self.count.sum()
        variance = self.square_sum.sum(0) / total - (self.sum.sum(0) / total).square()
        variance_floor = (variance * 1e-4).clamp_min(1e-10)
        count = self.count.clamp_min(1)
        mean = self.sum / count[:, None]
        variance = torch.maximum(self.square_sum / count[:, None] - mean.square(), variance_floor)
        log_prior = (count / total).log()
        distance = ((values[:, None, :] - mean).square() / variance).sum(-1)
        log_density = log_prior - 0.5 * (distance + variance.log().sum(-1))
        log_density[:, ~present] = -torch.inf
        return [self.labels[index] for index in log_density.argmax(-1).tolist()]


def feature_tensor(chunk, device):
    features = chunk[list(FEATURES)].to_numpy(dtype="float64")
    values = torch.as_tensor(features, device=device)
    if not torch.isfinite(values).all().item():
        raise ValueError("Non-finite pulse features in baseline input")
    return values


def fit(path, device):
    classifiers = {
        "target_class": DiagonalGaussian(CLASSES, device),
        "target_state": DiagonalGaussian(STATES, device),
    }
    for chunk in pd.read_csv(
        path, usecols=list(FEATURES) + list(classifiers), chunksize=CHUNK_SIZE,
        dtype={label: "string" for label in classifiers},
    ):
        values = feature_tensor(chunk, device)
        for label, classifier in classifiers.items():
            classifier.update(values, chunk[label])
    if not all(classifier.count.sum().item() for classifier in classifiers.values()):
        raise ValueError("Empty training set")
    return classifiers


def write_predictions(path, dest, classifiers, device, use_gaussian):
    dest.mkdir(parents=True)
    objects = set()
    majorities = {label: classifier.majority() for label, classifier in classifiers.items()}
    with (dest / "pulse_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        for index, chunk in enumerate(pd.read_csv(
            path, usecols=list(PULSE_KEY + FEATURES), chunksize=CHUNK_SIZE,
            dtype={column: "string" for column in PULSE_KEY}, keep_default_na=False,
        )):
            if use_gaussian:
                values = feature_tensor(chunk, device)
            for label, classifier in classifiers.items():
                chunk[label] = classifier.predict(values) if use_gaussian else majorities[label]
            chunk["local_cluster_id"] = chunk["target_class"]
            chunk["object_id"] = chunk["target_class"]
            chunk[list(PREDICTIONS)].to_csv(stream, index=False, header=index == 0)
            objects.update(chunk[["scenario_id", "data_version", "object_id"]]
                           .drop_duplicates().itertuples(index=False, name=None))

    with (dest / "object_track_links.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(LINKS)
        writer.writerows((*key, "", "no_match") for key in sorted(objects))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir must be a new directory")
    try:
        device = cuda_device()
        classifiers = fit(args.holdout / "train" / "A.csv", device)
        validation = args.holdout / "input" / "A.csv"
        args.out_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".b04-baseline-", dir=args.out_dir.parent) as scratch:
            staging = Path(scratch) / "predictions"
            write_predictions(validation, staging / "majority", classifiers, device, False)
            write_predictions(validation, staging / "gaussian", classifiers, device, True)
            staging.rename(args.out_dir)
        print(args.out_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Reference baseline failed: {error}\n")


if __name__ == "__main__":
    main()
