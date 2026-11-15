"""Load audited per-configuration runtime results for cross-method tables.

This module validates existing result.json/table_row.csv pairs.  It never
recomputes experiments, reads raw logs, selects configurations, or renders a
table.  Paths and recorded source-log identities stay inside the artifact.
"""

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import re

import analysis_common as base
import analyze_slide_mp as naming


_RECORD_TYPES = {
    "tennor_configuration_result": "TENNOR",
    "slide_oram_configuration_result": "SLIDE-ORAM",
}
_TIMING_BASES = {
    "TENNOR": "printed_batch_timer_with_rehash_amortization",
    "SLIDE-ORAM": "consecutive_batch_start_timestamps_with_rehash_amortization",
}
_COUNTS = ("batchsize", "n_runs", "n_batches", "n_inputs", "batches_per_epoch")
_UNKNOWN_THREADS = {"", "unknown", "none", "n/a", "na", "null"}


def _method(value):
    token = re.sub(r"[\s_+\-]", "", str(value)).casefold()
    if token == "tennor":
        return "TENNOR"
    if token in ("slideoram", "oram"):
        return "SLIDE-ORAM"
    raise ValueError(f"Unknown runtime method: {value!r}")


def _number(value, name, positive=False):
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number: {value!r}") from error
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}: {value!r}")
    return number


def _integer(value, name, minimum=1):
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError(f"{name} must be an integer >= {minimum}: {value!r}") from error
    if not number.is_finite() or number != number.to_integral_value() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}: {value!r}")
    return int(number)


def _threads(value):
    if value is None or str(value).strip().casefold() in _UNKNOWN_THREADS:
        return None
    return _integer(value, "threads")


def _close(actual, expected, label):
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-12):
        raise ValueError(f"{label} is inconsistent: {actual!r} vs {expected!r}")


def _parameters(raw):
    if not isinstance(raw, dict):
        raise ValueError("parameters must be an object")
    normalized = {}
    for key, value in raw.items():
        # Missing task parameters stay missing so the table builder can mark
        # compatibility as unverified instead of silently matching unknowns.
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, (dict, list, tuple, bool)):
            raise ValueError(f"parameters.{key} must be a scalar configuration value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"parameters.{key} is not finite")
        naming.add_parameter(normalized, key, value, "result.json parameters")
    return dict(sorted(normalized.items()))


def _source_path(raw, label):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must identify a source file inside the artifact")
    # This validates containment without requiring the original log to exist.
    return base.relative(base.local_path(raw.replace("\\", "/")))


def _row_csv(json_row, csv_raw):
    reader = csv.DictReader(io.StringIO(csv_raw, newline=""))
    columns = reader.fieldnames
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("table_row.csv must have unique named columns")
    rows = list(reader)
    if len(rows) != 1:
        raise ValueError("table_row.csv must contain exactly one data row")
    csv_row = rows[0]
    if None in csv_row or any(value is None for value in csv_row.values()):
        raise ValueError("table_row.csv has a row whose width differs from its header")
    if set(csv_row) != set(json_row):
        raise ValueError("table_row.csv columns must match result.json row fields")
    for key, raw in csv_row.items():
        expected = json_row[key]
        if key == "threads":
            equal = _threads(raw) == _threads(expected)
        elif key == "dataset":
            equal = naming.normalize_dataset(raw) == naming.normalize_dataset(expected)
        elif key == "method":
            equal = _method(raw) == _method(expected)
        elif expected is None:
            equal = raw.strip() == ""
        elif key in _COUNTS:
            equal = _integer(raw, f"CSV {key}") == _integer(expected, f"JSON row.{key}")
        elif key.endswith(("_seconds", "_hours")) or key == "speedup":
            _close(_number(raw, f"CSV {key}"), _number(expected, f"JSON row.{key}"), f"CSV/JSON {key}")
            equal = True
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            _close(_number(raw, f"CSV {key}"), _number(expected, f"JSON row.{key}"), f"CSV/JSON {key}")
            equal = True
        elif isinstance(expected, (dict, list)):
            raise ValueError(f"row.{key} must be a scalar value")
        else:
            equal = raw == str(expected)
        if not equal:
            raise ValueError(f"CSV/JSON {key} is inconsistent: {raw!r} vs {expected!r}")


def _measurement(document, summary, method, counts, window):
    runs = document.get("runs")
    selected_runs = summary.get("run_summaries")
    if not isinstance(runs, list) or not isinstance(selected_runs, list) or not selected_runs:
        raise ValueError("Formal results require runs and summary.run_summaries provenance")
    known = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("runs entries must be objects")
        run_id = _source_path(run.get("run_id"), "runs.run_id")
        if run_id in known:
            raise ValueError(f"Duplicate source run in result.json: {run_id}")
        digest = run.get("log_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"Missing or invalid source log_sha256: {run_id}")
        known[run_id] = digest.lower()
    evidence, seen, selected_count = [], set(), 0
    for selected in selected_runs:
        if not isinstance(selected, dict):
            raise ValueError("summary.run_summaries entries must be objects")
        run_id = _source_path(selected.get("run_id"), "summary.run_summaries.run_id")
        if run_id not in known:
            raise ValueError(f"Selected run has no source log checksum: {run_id}")
        if run_id in seen:
            raise ValueError(f"Selected run is repeated in summary: {run_id}")
        seen.add(run_id)
        selected_batches = selected.get("selected_batches")
        if not isinstance(selected_batches, list) or not selected_batches:
            raise ValueError(f"Selected batch identities are required for {run_id}")
        batches, batch_ids = [], set()
        for batch in selected_batches:
            if not isinstance(batch, dict):
                raise ValueError("selected_batches entries must be objects")
            batch_id = _integer(batch.get("batch"), "selected batch ID", minimum=0)
            header = _integer(batch.get("header_line"), "selected batch header_line")
            if batch_id in batch_ids:
                raise ValueError(f"Repeated selected global batch {batch_id} in {run_id}")
            batch_ids.add(batch_id)
            batches.append({"batch": batch_id, "header_line": header})
        if "n_batches" in selected and _integer(selected["n_batches"], "run summary n_batches") != len(batches):
            raise ValueError(f"Selected batch count is inconsistent for {run_id}")
        selected_count += len(batches)
        evidence.append({"run_id": run_id, "log_sha256": known[run_id],
                         "selected_batches": sorted(batches, key=lambda item: (item["batch"], item["header_line"]))})
    if len(evidence) != counts["n_runs"] or selected_count != counts["n_batches"]:
        raise ValueError("Selected source evidence disagrees with n_runs/n_batches")
    calibration = {key: summary.get(key) for key in (
        "rehash_calibration_source", "rehash_event_mean_seconds", "rehash_event_count",
        "rehash_interval_batches")}
    if calibration["rehash_event_mean_seconds"] is not None:
        calibration["rehash_event_mean_seconds"] = _number(
            calibration["rehash_event_mean_seconds"], "summary.rehash_event_mean_seconds")
    for key, minimum in (("rehash_event_count", 0), ("rehash_interval_batches", 1)):
        if calibration[key] is not None:
            calibration[key] = _integer(calibration[key], f"summary.{key}", minimum=minimum)
    raw_events = summary.get("rehash_events", [])
    if not isinstance(raw_events, list):
        raise ValueError("summary.rehash_events must be an array")
    calibration["events"] = []
    for event in raw_events:
        if not isinstance(event, dict):
            raise ValueError("summary.rehash_events entries must be objects")
        copied = dict(event)
        if "run_id" in copied:
            copied["run_id"] = _source_path(copied["run_id"], "rehash_events.run_id")
        if "seconds" in copied:
            copied["seconds"] = _number(copied["seconds"], "rehash_events.seconds")
        calibration["events"].append(copied)
    calibration["events"].sort(key=lambda item: json.dumps(item, sort_keys=True))
    return json.dumps({"method": method, "runs": sorted(evidence, key=lambda item: item["run_id"]),
                       "window": window, "rehash_calibration": calibration},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number is not allowed: {value}")


def load_result(path, expected_method=None):
    """Validate one formal result pair and return normalized runtime metadata.

    ``path`` accepts a directory, result.json, or table_row.csv.  Seconds and
    hours are explicit field units and must agree when both are present.
    Actual threads come only from row.threads; a configured OMP count remains
    a parameter and is never substituted for an unknown actual count.
    """
    requested = base.local_path(path)
    if requested.is_dir():
        directory = requested
    elif requested.name in ("result.json", "table_row.csv"):
        directory = requested.parent
    else:
        raise ValueError("Expected a result directory, result.json, or table_row.csv")
    result_path = base.local_path(directory / "result.json")
    csv_path = base.local_path(directory / "table_row.csv")
    if not result_path.is_file() or not csv_path.is_file():
        raise ValueError(f"Both result.json and table_row.csv are required: {directory}")
    json_bytes, csv_bytes = result_path.read_bytes(), csv_path.read_bytes()
    try:
        document = json.loads(json_bytes.decode("utf-8-sig"), parse_constant=_reject_constant)
        csv_text = csv_bytes.decode("utf-8-sig")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read result pair {directory}: {error}") from error
    if not isinstance(document, dict) or document.get("record_type") not in _RECORD_TYPES:
        raise ValueError("Only formal TENNOR/SLIDE-ORAM configuration results are accepted; samples-only records are not")
    method = _RECORD_TYPES[document["record_type"]]
    if expected_method is not None and _method(expected_method) != method:
        raise ValueError(f"Expected {_method(expected_method)}, found {method}")
    row, summary = document.get("row"), document.get("summary")
    if not isinstance(row, dict) or not isinstance(summary, dict):
        raise ValueError("Formal results require row and summary objects")
    if _method(row.get("method")) != method:
        raise ValueError("row.method disagrees with record_type")
    if "component" in document and _method(document["component"]) != method:
        raise ValueError("component disagrees with record_type")
    _row_csv(row, csv_text)
    timing_basis = row.get("timing_basis")
    if timing_basis != _TIMING_BASES[method]:
        raise ValueError(f"Unsupported {method} timing_basis: {timing_basis!r}")
    dataset = naming.normalize_dataset(row.get("dataset"))
    counts = {key: _integer(row.get(key), f"row.{key}") for key in _COUNTS}
    parameters = _parameters(document.get("parameters"))
    if _integer(parameters.get("Batchsize"), "parameters.Batchsize") != counts["batchsize"]:
        raise ValueError("row.batchsize and parameters.Batchsize disagree")
    records = _integer(parameters.get("totRecords"), "parameters.totRecords")
    if counts["n_inputs"] != counts["n_batches"] * counts["batchsize"]:
        raise ValueError("n_inputs must equal n_batches * batchsize")
    if counts["n_runs"] > counts["n_batches"]:
        raise ValueError("n_runs cannot exceed n_batches")
    if counts["batches_per_epoch"] != records // counts["batchsize"]:
        raise ValueError("batches_per_epoch must equal floor(totRecords / batchsize)")
    seconds = _number(row["batch_seconds"], "row.batch_seconds", positive=True) if "batch_seconds" in row else None
    hours = _number(row["batch_hours"], "row.batch_hours", positive=True) if "batch_hours" in row else None
    if seconds is None and hours is None:
        raise ValueError("Batch timing requires an explicit batch_seconds or batch_hours unit")
    if seconds is None:
        seconds = _number(hours * 3600, "converted batch_seconds", positive=True)
    elif hours is not None:
        _close(seconds, hours * 3600, "batch_seconds/batch_hours")
    hours = _number(seconds / 3600, "converted batch_hours", positive=True)
    epoch_hours = _number(row.get("epoch_hours"), "row.epoch_hours", positive=True)
    _close(epoch_hours, counts["batches_per_epoch"] * seconds / 3600, "epoch_hours")
    expected = dict(counts, batch_seconds=seconds, batch_hours=hours, epoch_hours=epoch_hours)
    for key, value in expected.items():
        if key in summary:
            if key in _COUNTS:
                if _integer(summary[key], f"summary.{key}") != value:
                    raise ValueError(f"summary/row {key} is inconsistent")
            else:
                _close(_number(summary[key], f"summary.{key}"), value, f"summary/row {key}")
    window = summary.get("window")
    if window is not None and not isinstance(window, dict):
        raise ValueError("summary.window must be an object or null")
    if window is not None:
        window = dict(window)
        if "skip_batches" in window:
            window["skip_batches"] = _integer(window["skip_batches"], "window.skip_batches", minimum=0)
        if window.get("max_batches") is not None:
            window["max_batches"] = _integer(window["max_batches"], "window.max_batches")
    measurement_key = _measurement(document, summary, method, counts, window)
    return dict(source=base.relative(result_path), csv_source=base.relative(csv_path),
                method=method, dataset=dataset, threads=_threads(row.get("threads")),
                **counts, batch_seconds=seconds, batch_hours=hours, epoch_hours=epoch_hours,
                parameters=parameters, window=window, measurement_key=measurement_key,
                timing_basis=timing_basis,
                source_checksums={"result_json": hashlib.sha256(json_bytes).hexdigest(),
                                  "table_row_csv": hashlib.sha256(csv_bytes).hexdigest()})
