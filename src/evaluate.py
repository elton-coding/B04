"""Compare B04 result CSVs with labeled observations using local metrics."""

import argparse
import csv
import json
import sqlite3
from contextlib import closing
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory


PULSE_KEY = ("scenario_id", "data_version", "pulse_id")
TRUTH_A = PULSE_KEY + ("platform_id", "target_id", "target_class", "target_state")
TRUTH_B = PULSE_KEY[:2] + ("track_id", "target_id")
PREDICTIONS = PULSE_KEY + ("local_cluster_id", "object_id", "target_class", "target_state")
LINKS = PULSE_KEY[:2] + ("object_id", "track_id", "link_status")
CLASSES = tuple(f"R{i:02}" for i in range(1, 9))
STATES = ("0", "1", "2")

SCHEMA = """
CREATE TABLE truth_a(s TEXT, v TEXT, pulse TEXT, platform TEXT, target TEXT, cls TEXT, state TEXT,
                     PRIMARY KEY(s,v,pulse)) WITHOUT ROWID;
CREATE TABLE tracks(s TEXT, v TEXT, track TEXT, target TEXT,
                    PRIMARY KEY(s,v,track)) WITHOUT ROWID;
CREATE INDEX tracks_target ON tracks(s,v,target);
CREATE TABLE predictions(s TEXT, v TEXT, pulse TEXT, local TEXT, object TEXT, cls TEXT, state TEXT,
                         PRIMARY KEY(s,v,pulse)) WITHOUT ROWID;
CREATE TABLE links(s TEXT, v TEXT, object TEXT, track TEXT, status TEXT,
                   PRIMARY KEY(s,v,object)) WITHOUT ROWID;
CREATE UNIQUE INDEX matched_track ON links(s,v,track) WHERE status='matched';
"""


def csv_rows(path, fields, *, exact=False, nullable=()):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or not set(fields) <= set(header):
            raise ValueError(f"{path}: expected columns {', '.join(fields)}")
        if exact and tuple(header) != fields:
            raise ValueError(f"{path}: expected column order {', '.join(fields)}")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"{path}:{reader.line_num}: invalid number of fields")
            values = tuple(row[field] for field in fields)
            if exact and any(value != value.strip() for value in values):
                raise ValueError(f"{path}:{reader.line_num}: unexpected outer whitespace")
            if not exact:
                values = tuple(
                    value.strip() if field in ("target_class", "target_state") else value
                    for field, value in zip(fields, values)
                )
            if any(not value for field, value in zip(fields, values) if field not in nullable):
                raise ValueError(f"{path}:{reader.line_num}: required field is empty")
            if fields in (TRUTH_A, PREDICTIONS):
                if values[-2] not in CLASSES or values[-1] not in STATES:
                    raise ValueError(f"{path}:{reader.line_num}: invalid target_class or target_state")
            if fields == LINKS:
                track, status = values[-2:]
                if status not in ("matched", "no_match") or bool(track) != (status == "matched"):
                    raise ValueError(f"{path}:{reader.line_num}: invalid track_id / link_status")
            yield values


def load(db, path, fields, statement, **options):
    try:
        for batch in batched(csv_rows(path, fields, **options), 16_384):
            db.executemany(statement, batch)
    except sqlite3.IntegrityError as error:
        raise ValueError(f"{path}: duplicate pulse, object, or matched track key") from error


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def scores(tp, predicted, actual):
    return {
        "precision": divide(tp, predicted),
        "recall": divide(tp, actual),
        "f1": divide(2 * tp, predicted + actual),
        "true_positives": tp,
        "predicted": predicted,
        "actual": actual,
    }


def clustering(db, table):
    # Pair counts are computed from grouped sizes, without enumerating pulse pairs.
    n, p, r = db.execute(f"""
        SELECT SUM(c.n), SUM(1.0*c.n*c.n/p.n), SUM(1.0*c.n*c.n/t.n)
        FROM {table} c
        JOIN {table}_pred p USING(s,v,scope,pred)
        JOIN {table}_truth t USING(s,v,scope,target)
    """).fetchone()
    p, r = p / n, r / n
    return {"b3_precision": p, "b3_recall": r, "b3_f1": 2*p*r/(p+r)}


def group_members(db, table, scope, prediction):
    # scope and prediction are fixed SQL expressions supplied by evaluate().
    db.executescript(f"""
        CREATE TABLE {table} AS
        SELECT a.s, a.v, {scope} AS scope, a.target, {prediction} AS pred, COUNT(*) AS n
        FROM truth_a a JOIN predictions p USING(s,v,pulse)
        GROUP BY a.s,a.v,scope,a.target,pred;
        CREATE TABLE {table}_pred AS SELECT s,v,scope,pred,SUM(n) AS n
            FROM {table} GROUP BY s,v,scope,pred;
        CREATE UNIQUE INDEX {table}_pred_key ON {table}_pred(s,v,scope,pred);
        CREATE TABLE {table}_truth AS SELECT s,v,scope,target,SUM(n) AS n
            FROM {table} GROUP BY s,v,scope,target;
        CREATE UNIQUE INDEX {table}_truth_key ON {table}_truth(s,v,scope,target);
    """)


def cross_platform_pairs(db):
    def pair_count(table):
        return db.execute(f"SELECT COALESCE(SUM(n*(n-1)/2),0) FROM {table}").fetchone()[0]

    difference = [
        pair_count("fused" + suffix) - pair_count("by_platform" + suffix)
        for suffix in ("", "_pred", "_truth")
    ]
    return scores(*difference)


def classification(db, field, labels):
    rows = db.execute(f"""
        SELECT a.{field},p.{field},COUNT(*)
        FROM truth_a a JOIN predictions p USING(s,v,pulse)
        GROUP BY a.{field},p.{field}
    """).fetchall()
    counts = {(truth, prediction): count for truth, prediction, count in rows}
    by_label = {
        label: scores(
            counts.get((label, label), 0),
            sum(counts.get((other, label), 0) for other in labels),
            sum(counts.get((label, other), 0) for other in labels),
        )
        for label in labels
    }
    present_f1 = [result["f1"] for result in by_label.values() if result["actual"]]
    return {
        "accuracy": sum(counts.get((label, label), 0) for label in labels) / sum(counts.values()),
        "macro_f1_on_present_labels": sum(present_f1) / len(present_f1),
        "by_label": by_label,
        "confusion_matrix": {truth: {pred: counts.get((truth, pred), 0) for pred in labels}
                             for truth in labels},
    }


def association(db):
    # An IoU strictly greater than 0.5 makes this alignment unique on both sides.
    db.executescript("""
        CREATE TABLE aligned AS
        SELECT c.s,c.v,c.target,c.pred
        FROM fused c
        JOIN fused_pred p USING(s,v,scope,pred)
        JOIN fused_truth t USING(s,v,scope,target)
        WHERE 3*c.n > p.n+t.n;
    """)
    actual, unmatched = db.execute("""
        SELECT COUNT(*), SUM(NOT EXISTS(
            SELECT 1 FROM tracks b WHERE b.s=t.s AND b.v=t.v AND b.target=t.target))
        FROM fused_truth t
    """).fetchone()
    tp, correct_no_match = db.execute("""
        SELECT
            COALESCE(SUM(l.status='matched' AND b.target=a.target),0),
            COALESCE(SUM(l.status='no_match' AND NOT EXISTS(
                SELECT 1 FROM tracks any_b
                WHERE any_b.s=a.s AND any_b.v=a.v AND any_b.target=a.target)),0)
        FROM aligned a JOIN links l ON l.s=a.s AND l.v=a.v AND l.object=a.pred
        LEFT JOIN tracks b ON b.s=l.s AND b.v=l.v AND b.track=l.track
    """).fetchone()
    predicted = db.execute("SELECT COUNT(*) FROM links WHERE status='matched'").fetchone()[0]
    aligned = db.execute("SELECT COUNT(*) FROM aligned").fetchone()[0]
    return {
        "matched_links": scores(tp, predicted, actual - unmatched),
        "correct_no_match": correct_no_match,
        "truth_no_match": unmatched,
        "no_match_recall": divide(correct_no_match, unmatched),
        "aligned_objects": aligned,
        "truth_objects": actual,
        "alignment_iou_strictly_above": 0.5,
    }


def evaluate(source_a, source_b, predictions, links, *, labeled=True):
    # Keep the multi-million-row join in a temporary SQLite file.
    with TemporaryDirectory(prefix="b04-evaluation-") as scratch:
        with closing(sqlite3.connect(Path(scratch) / "evaluation.sqlite3")) as db:
            db.executescript("PRAGMA journal_mode=OFF; PRAGMA temp_store=FILE;" + SCHEMA)
            if labeled:
                load(db, source_a, TRUTH_A, "INSERT INTO truth_a VALUES(?,?,?,?,?,?,?)")
                load(db, source_b, TRUTH_B, """
                    INSERT INTO tracks VALUES(?,?,?,?) ON CONFLICT(s,v,track) DO UPDATE
                    SET target=CASE WHEN excluded.target=tracks.target THEN tracks.target ELSE NULL END
                """)
                if db.execute("SELECT 1 FROM tracks WHERE target IS NULL LIMIT 1").fetchone():
                    raise ValueError("B truth assigns inconsistent target IDs to one track")
            else:
                load(db, source_a, TRUTH_A[:4], "INSERT INTO truth_a(s,v,pulse,platform) VALUES(?,?,?,?)")
                load(db, source_b, TRUTH_B[:3], "INSERT OR IGNORE INTO tracks(s,v,track) VALUES(?,?,?)")
            load(db, predictions, PREDICTIONS, "INSERT INTO predictions VALUES(?,?,?,?,?,?,?)", exact=True)
            load(db, links, LINKS, "INSERT INTO links VALUES(?,?,?,?,?)", exact=True, nullable=("track_id",))
            db.commit()

            checks = (
                ("SELECT 1 FROM truth_a LIMIT 1", "A observations have no pulses", False),
                ("SELECT 1 FROM truth_a a WHERE NOT EXISTS (SELECT 1 FROM predictions p WHERE (p.s,p.v,p.pulse)=(a.s,a.v,a.pulse)) LIMIT 1",
                 "A observations include pulses without predictions", True),
                ("SELECT 1 FROM predictions p WHERE NOT EXISTS (SELECT 1 FROM truth_a a WHERE (p.s,p.v,p.pulse)=(a.s,a.v,a.pulse)) LIMIT 1",
                 "Predictions include pulses outside the evaluation set", True),
                ("SELECT 1 FROM links l WHERE l.status='matched' AND NOT EXISTS (SELECT 1 FROM tracks b WHERE (b.s,b.v,b.track)=(l.s,l.v,l.track)) LIMIT 1",
                 "A matched track is absent from B observations", True),
            )
            for query, message, invalid_if_found in checks:
                if bool(db.execute(query).fetchone()) == invalid_if_found:
                    raise ValueError(message)

            group_members(db, "fused", "''", "p.object")
            for left, right in (("fused_pred", "links"), ("links", "fused_pred")):
                left_key = "pred" if left == "fused_pred" else "object"
                right_key = "pred" if right == "fused_pred" else "object"
                scope_filter = "AND y.scope=''" if right == "fused_pred" else ""
                if db.execute(f"""
                    SELECT 1 FROM {left} x WHERE NOT EXISTS (
                        SELECT 1 FROM {right} y
                        WHERE (x.s,x.v,x.{left_key})=(y.s,y.v,y.{right_key}) {scope_filter})
                    LIMIT 1
                """).fetchone():
                    raise ValueError("Links must contain exactly one row for each predicted object")
            if not labeled:
                return {
                    "method": "b04_structure_v1",
                    "pulse_count": db.execute("SELECT COUNT(*) FROM truth_a").fetchone()[0],
                    "object_count": db.execute("SELECT COUNT(*) FROM links").fetchone()[0],
                    "matched_tracks": db.execute("SELECT COUNT(*) FROM links WHERE status='matched'").fetchone()[0],
                }
            group_members(db, "local", "a.platform", "p.local")
            group_members(db, "by_platform", "a.platform", "p.object")

            return {
                "method": "b04_local_v1",
                "pulse_count": db.execute("SELECT COUNT(*) FROM truth_a").fetchone()[0],
                "local_clustering": clustering(db, "local"),
                "target_class": classification(db, "cls", CLASSES),
                "target_state": classification(db, "state", STATES),
                "cross_platform_fusion": {
                    **clustering(db, "fused"),
                    "cross_platform_pairwise": cross_platform_pairs(db),
                },
                "object_track_association": association(db),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("pulse-predictions", "object-track-links"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    for option in ("truth-a", "truth-b", "input-a", "input-b"):
        parser.add_argument(f"--{option}", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    labeled = bool(args.truth_a and args.truth_b and not (args.input_a or args.input_b))
    input_only = bool(args.input_a and args.input_b and not (args.truth_a or args.truth_b))
    if not (labeled or input_only):
        parser.error("provide --truth-a and --truth-b to score, or --input-a and --input-b to check format")
    try:
        report = evaluate(args.truth_a or args.input_a, args.truth_b or args.input_b,
                          args.pulse_predictions, args.object_track_links, labeled=labeled)
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8", newline="\n")
        else:
            print(text, end="")
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
