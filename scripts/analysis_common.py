#!/usr/bin/env python3
"""Shared artifact paths, metadata, accuracy parsing, averaging, and plotting helpers."""

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics


ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "wiki10": ("Wiki10-31K",),
    "wiki325": ("Wiki-325K",),
    "amazon670": ("Amazon-670K",),
    "amzprime": ("Amazon-Sub",),
}
ALIASES = {
    "wiki10": "wiki10", "wiki1031k": "wiki10",
    "wiki325": "wiki325", "wiki325k": "wiki325", "wikilshtc325k": "wiki325",
    "amazon": "amazon670", "amazon670": "amazon670", "amazon670k": "amazon670",
    "amz670": "amazon670", "amzprime": "amzprime", "amazonsub": "amzprime",
    "amzsub": "amzprime",
}
COMPARISON_KEYS = """RangePow K L Sparsity Batchsize Rehash Rebuild InputDim
totRecords totRecordsTest BUCKETSIZE binsize isBruteForce NNSorMPS BFnumber
isMPWTA isRandom Lr sizesOfLayers numLayer""".split()
EPOCH = re.compile(r"Epoch\s+(\d+)\s*$")


def local_path(path):
    """All inputs/outputs stay inside the simulated artifact, including symlinks."""
    path = Path(path)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path must remain inside {ROOT}: {path}")
    return path


def relative(path):
    return local_path(path).relative_to(ROOT).as_posix()


def sha256(path):
    return hashlib.sha256(local_path(path).read_bytes()).hexdigest()


def tail_epoch(metadata):
    """A successful launcher record confirms only its configured final epoch."""
    if metadata.get("status") != "completed" or metadata.get("returncode") != 0:
        return None
    raw = str(metadata.get("effective_values", {}).get("Epoch", ""))
    if re.fullmatch(r"[1-9][0-9]*", raw):
        return int(raw) - 1
    return None


def parse_log(text, completed_epoch=None, *, epoch_batches=None, configured_epochs=None):
    """Return latest complete epoch rows plus a disposition for every block.

    A strictly later header confirms a nonterminal block. Repeated/backward
    headers are restart boundaries, not completion evidence. An unconfirmed
    terminal block is discarded without confirmation. For raw upstream WTA
    logs, the optional CSV-derived epoch size confirms an epoch-end evaluation
    only at its exact iteration boundary; this does not confirm process exit.
    """
    if epoch_batches is not None:
        if (not isinstance(epoch_batches, int) or epoch_batches <= 0
                or not isinstance(configured_epochs, int) or configured_epochs <= 0):
            raise ValueError("Epoch-boundary parsing requires positive epoch_batches and configured_epochs")
    blocks = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().lstrip("\ufeff")
        if not line:
            continue
        match = EPOCH.fullmatch(line)
        if match:
            blocks.append({"epoch": int(match[1]), "header_line": line_number,
                           "occurrence": len(blocks) + 1, "evaluations": 0,
                           "last": None, "invalid_lines": []})
            continue
        if not blocks:
            raise ValueError(f"Line {line_number}: expected an Epoch header, got {line!r}")
        block = blocks[-1]
        fields = line.split()
        try:
            if len(fields) != 3 or not re.fullmatch(r"[0-9]+", fields[0]):
                raise ValueError("expected batch-count, time, P@1")
            batches, training_time, accuracy = int(fields[0]), float(fields[1]), float(fields[2])
            if not math.isfinite(training_time) or training_time < 0:
                raise ValueError("invalid cumulative training time")
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("P@1 must be finite and between 0 and 1")
            previous = block["last"]
            if previous and (batches < previous["batches"] or training_time < previous["training_time"]):
                raise ValueError("counters decrease within an epoch block")
            block["last"] = {"batches": batches, "training_time": training_time,
                             "p_at_1": accuracy, "metric_line": line_number}
            block["evaluations"] += 1
        except ValueError as error:
            # In particular, never fall back to a mid-epoch result after a torn
            # final record. Keep the rejection visible in the audit output.
            block["invalid_lines"].append({"line": line_number, "error": str(error)})

    selected = {}
    audit = []
    for index, block in enumerate(blocks):
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if block["invalid_lines"]:
            reason = "invalid_record"
        elif block["last"] is None:
            reason = "empty_block"
        elif epoch_batches is not None:
            if block["epoch"] >= configured_epochs:
                reason = "epoch_outside_configuration"
            elif block["last"]["batches"] != (block["epoch"] + 1) * epoch_batches:
                reason = "missing_epoch_end_iteration"
            else:
                reason = "epoch_end_iteration"
        elif following is not None:
            reason = ("followed_by_later_epoch" if following["epoch"] > block["epoch"]
                      else "restart_without_advance")
        else:
            reason = ("completed_run_tail" if block["epoch"] == completed_epoch
                      else "unconfirmed_tail")
        keep = reason in {"followed_by_later_epoch", "completed_run_tail", "epoch_end_iteration"}
        entry = {key: block[key] for key in
                 ("epoch", "header_line", "occurrence", "evaluations", "invalid_lines")}
        entry.update(disposition="kept" if keep else "discarded", reason=reason)
        if block["last"]:
            entry.update(block["last"])
        audit.append(entry)
        if keep:
            old = selected.get(block["epoch"])
            if old is not None:
                audit[old]["disposition"] = "discarded"
                audit[old]["reason"] = "superseded_by_later_complete_occurrence"
            selected[block["epoch"]] = index
    rows = [{key: audit[index][key] for key in
             ("epoch", "batches", "training_time", "p_at_1", "header_line",
              "metric_line", "occurrence")}
            for _, index in sorted(selected.items())]
    return rows, audit


def aggregate(runs):
    grouped = {}
    seen = set()
    for run in runs:
        for row in run["rows"]:
            identity = (run["dataset"], run["run_id"], row["epoch"])
            if identity in seen:
                raise ValueError(f"Duplicate run/epoch sample: {identity}")
            seen.add(identity)
            grouped.setdefault((run["dataset"], row["epoch"]), []).append(
                (run["run_id"], row["p_at_1"]))
    result = []
    for (dataset, epoch), samples in sorted(grouped.items()):
        values = [value for _, value in samples]
        result.append({"dataset": dataset, "epoch": epoch,
                       "p_at_1_mean": statistics.mean(values),
                       "p_at_1_std": statistics.stdev(values) if len(values) > 1 else None,
                       "n_runs": len(values), "run_ids": [name for name, _ in samples]})
    return result


def read_metadata(directory):
    path = local_path(directory / "run.json")
    if not path.is_file():
        return {}, None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("effective_values", {}), dict):
        raise ValueError(f"Invalid run metadata: {relative(path)}")
    return value, path


def portable_path(raw):
    # Recorded Linux paths are provenance, never locations to read on this host.
    return PurePosixPath(str(raw).replace("\\", "/"))


def locate_log(directory, metadata):
    stored = metadata.get("effective_values", {}).get("logFile")
    if stored:
        log = portable_path(stored)
        origin = portable_path(metadata.get("cwd", "")).parent
        try:
            suffix = log.relative_to(origin)
        except ValueError:
            suffix = log if not log.is_absolute() and ":" not in str(log) else None
        if suffix is not None:
            candidate = local_path(directory / str(suffix))
            if directory not in candidate.parents:
                raise ValueError(f"logFile must remain inside its run directory: {stored}")
            if candidate.is_file():
                return candidate
        # Default logs are portable even when metadata lacks the original cwd.
        if log.name != "training.log":
            raise FileNotFoundError(f"Cannot locate custom logFile for {relative(directory)}: {stored}")
    path = local_path(directory / "training.log")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {relative(path)}")
    return path


def dataset_from_metadata(metadata):
    values = metadata.get("effective_values", {})
    candidates = [values.get("trainData", "")]
    candidates += [item.get("path", "") for item in metadata.get("inputs", [])]
    for candidate in candidates:
        folder = portable_path(candidate).parent.name
        key = re.sub(r"[^a-z0-9]", "", folder.lower())
        if key in ALIASES:
            return ALIASES[key]
    name = portable_path(metadata.get("config", "")).stem.lower()
    for token in name.split("_"):
        key = re.sub(r"[^a-z0-9]", "", token)
        if key in ALIASES:
            return ALIASES[key]
    return None


def load_runs(paths, dataset_filter=None):
    automatic = not paths
    if automatic:
        runs_dir = local_path("SLIDE-MP/dataset/runs")
        paths = sorted(path for path in runs_dir.glob("*") if path.is_dir()
                       and ((path / "training.log").is_file() or (path / "run.json").is_file()))
    runs, skipped, seen = [], [], set()
    for raw in paths:
        path = local_path(raw)
        if not path.exists():
            raise ValueError(f"Input does not exist: {relative(path)}")
        directory = path if path.is_dir() else path.parent
        metadata, manifest = read_metadata(directory)
        values = metadata.get("effective_values", {})
        if (str(values.get("isMPWTA", "1")) != "1"
                or str(values.get("isBruteForce", "0")) != "0"):
            skipped.append({"input": relative(path), "reason": "not an MP-WTA sparse run"})
            continue
        dataset = dataset_from_metadata(metadata)
        if dataset_filter and dataset and dataset != dataset_filter:
            skipped.append({"input": relative(path), "reason": "dataset filter"})
            continue
        dataset = dataset or dataset_filter
        if dataset is None:
            raise ValueError(f"Cannot identify dataset for {relative(path)}; use --dataset.")
        try:
            declared_log = locate_log(directory, metadata)
        except FileNotFoundError as error:
            if path.is_file():
                declared_log = None
            elif not automatic:
                raise
            else:
                skipped.append({"input": relative(path), "reason": str(error)})
                continue
        log = declared_log if path.is_dir() else path
        if log in seen:
            continue
        seen.add(log)
        confirmed_tail = tail_epoch(metadata) if log == declared_log else None
        rows, audit = parse_log(log.read_text(encoding="utf-8-sig"), confirmed_tail)
        runs.append({"run_id": relative(log), "dataset": dataset,
                     "log_sha256": sha256(log), "metadata": relative(manifest) if manifest else None,
                     "metadata_sha256": sha256(manifest) if manifest else None,
                     "status": metadata.get("status", "unknown"),
                     "confirmed_terminal_epoch": confirmed_tail,
                     "effective_values": values, "rows": rows, "audit": audit})
    return runs, skipped


def validate_comparable(runs):
    """Do not silently average different model/hashing configurations."""
    for dataset in DATASETS:
        group = [run for run in runs if run["dataset"] == dataset and run["rows"]]
        for key in COMPARISON_KEYS:
            values = {re.sub(r"\s+", "", str(run["effective_values"][key]))
                      for run in group if key in run["effective_values"]}
            if len(values) > 1:
                raise ValueError(f"{dataset}: runs disagree on {key}: {sorted(values)}. "
                                 "Select comparable run directories explicitly.")




def write_csv(path, columns, rows):
    with local_path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list)
                             else value for key, value in row.items()})


def make_plot(dataset, summary, reference, output, *, wta_summary=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    def plot_measurements(ax, rows, label, color, marker):
        counts = [row["n_runs"] for row in rows]
        sample_label = (str(min(counts)) if min(counts) == max(counts)
                        else f"{min(counts)}–{max(counts)}")
        # Each measured curve breaks at its own missing epochs.
        measured = {row["epoch"]: row for row in rows}
        epochs = list(range(min(measured), max(measured) + 1))
        means = [measured[e]["p_at_1_mean"] if e in measured else math.nan for e in epochs]
        deviations = [(measured[e]["p_at_1_std"] if measured[e]["p_at_1_std"] is not None
                       else math.nan) if e in measured else math.nan for e in epochs]
        ax.plot(epochs, means, color=color, marker=marker, markersize=3,
                label=f"{label} (mean, n={sample_label})")
        if any(count > 1 for count in counts):
            ax.fill_between(epochs, [a - b for a, b in zip(means, deviations)],
                            [a + b for a, b in zip(means, deviations)], color=color,
                            alpha=0.18, label=f"{label} ±1 sample SD")
    with plt.rc_context({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
                         "pdf.fonttype": 42, "axes.linewidth": 1.1}):
        fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
        if wta_summary:
            plot_measurements(ax, wta_summary, "WTA 50HT logs", "#1f77b4", "s")
        elif reference:
            ax.plot([row["epoch"] for row in reference], [row["p_at_1"] for row in reference],
                    color="#1f77b4", linestyle="--", label="WTA 50HT (historical reference)")
        plot_measurements(ax, summary, "MP-WTA", "#ff7f0e", "o")
        title = DATASETS[dataset][0]
        ax.set(xlabel="Epoch", ylabel="Accuracy (P@1)", title=title)
        if wta_summary and any(row.get("evaluation_scope") == "first_50_test_batches"
                               for row in wta_summary):
            ax.text(0.02, 0.98, "WTA before final epoch: first 50 test batches\n"
                    "MP-WTA: all complete test batches", transform=ax.transAxes,
                    va="top", fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)
        for extension in ("pdf", "png"):
            fig.savefig(local_path(output / f"{dataset}_epoch_accuracy.{extension}"), dpi=180)
        plt.close(fig)
    return matplotlib.__version__
