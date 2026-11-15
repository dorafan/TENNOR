"""Read independent upstream SLIDE WTA logs with their recorded CSV settings."""

from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import PurePosixPath
import re

import analysis_common as base


DEFAULTS = {
    "wiki10": ("Wiki10", "wiki10"),
    "amzprime": ("AmzPrime", "amzprime"),
    "wiki325": ("Wiki325", "wiki325"),
    "amazon670": ("Amazon", "amz"),
}
PATH_KEYS = {"trainData", "testData", "weight", "savedweight", "logFile"}
ALIASES = dict(base.ALIASES, wiki31k="wiki10", wikilshtc="wiki325",
               wikilshtc325="wiki325", amz="amazon670", amz670k="amazon670",
               amazonprime="amzprime")


def _dataset(value):
    return ALIASES.get(re.sub(r"[\s_-]+", "", value).casefold())


def _config_dataset(path, values):
    """Treat recognizable dataset names as evidence, including archived CSVs."""
    evidence = set()
    for key in ("trainData", "testData"):
        parent = PurePosixPath(values.get(key, "").replace("\\", "/")).parent.name
        identity = _dataset(parent)
        if identity:
            evidence.add(identity)
    stem = re.sub(r"^config[\s_-]+", "", path.stem, flags=re.IGNORECASE)
    for alias, identity in ALIASES.items():
        pattern = r"[\s_-]*".join(re.escape(character) for character in alias)
        if re.match(pattern + r"(?=$|[\s_-])", stem, re.IGNORECASE):
            evidence.add(identity)
    if len(evidence) > 1:
        raise ValueError(f"{base.relative(path)}: conflicting dataset identities: {sorted(evidence)}")
    return next(iter(evidence), None)


def _positive_integer(values, key, source):
    value = values.get(key, "")
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"{source}: {key} must be a positive integer")
    return int(value)


def _integer_vector(values, key, length, source, minimum=1):
    parts = values.get(key, "").split(",")
    if len(parts) != length or any(not re.fullmatch(r"[0-9]+", part.strip()) for part in parts):
        raise ValueError(f"{source}: {key} must contain {length} integers")
    vector = [int(part.strip()) for part in parts]
    if any(value < minimum for value in vector):
        raise ValueError(f"{source}: every {key} value must be at least {minimum}")
    return vector


def _number(value, source):
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{source}: invalid number {value!r}") from error
    if not number.is_finite() or abs(number.adjusted()) > 100:
        raise ValueError(f"{source}: invalid or excessive number {value!r}")
    return number


def _normalize(value):
    parts = []
    for part in value.split(","):
        part = part.strip()
        try:
            number = Decimal(part)
        except InvalidOperation:
            parts.append(part.casefold())
            continue
        if not number.is_finite() or abs(number.adjusted()) > 100:
            raise ValueError(f"Invalid or excessive configuration value: {part!r}")
        parts.append("0" if not number else format(number.normalize(), "f"))
    return ",".join(parts)


def _read_config(path, dataset):
    source = base.relative(path)
    values = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            raise ValueError(f"{source}:{line_number}: expected KEY=VALUE")
        if key in values:
            raise ValueError(f"{source}:{line_number}: duplicate setting {key}")
        values[key] = value
    identity = _config_dataset(path, values)
    if identity and identity != dataset:
        raise ValueError(f"{source}: dataset {identity} conflicts with requested {dataset}")

    integers = {key: _positive_integer(values, key, source) for key in
                ("numLayer", "Batchsize", "InputDim", "totRecords", "totRecordsTest",
                 "Epoch", "Stepsize", "Rehash", "Rebuild")}
    layers = integers["numLayer"]
    _integer_vector(values, "sizesOfLayers", layers, source)
    _integer_vector(values, "K", layers, source)
    _integer_vector(values, "RangePow", layers, source, minimum=0)
    tables = _integer_vector(values, "L", layers, source)
    if tables[-1] != 50:
        raise ValueError(f"{source}: WTA 50HT requires output-layer L=50")
    sparsity = values.get("Sparsity", "").split(",")
    if len(sparsity) != 2 * layers:
        raise ValueError(f"{source}: Sparsity must contain {2 * layers} values")
    if any(not 0 <= _number(value.strip(), source) <= 1 for value in sparsity):
        raise ValueError(f"{source}: Sparsity values must be between 0 and 1")
    if _number(values.get("Lr", ""), source) <= 0:
        raise ValueError(f"{source}: Lr must be positive")
    for key in ("trainData", "testData"):
        if not values.get(key):
            raise ValueError(f"{source}: missing {key}")
    if integers["totRecords"] // integers["Batchsize"] == 0:
        raise ValueError(f"{source}: training data has no complete batch")
    if integers["totRecordsTest"] // integers["Batchsize"] == 0:
        raise ValueError(f"{source}: test data has no complete batch")
    parameters = {key: _normalize(value) for key, value in values.items() if key not in PATH_KEYS}
    return integers, parameters


def load_wta_runs(log_paths, config_paths, dataset, *, evaluation="full"):
    """Load WTA inputs; one CSV may describe several independent log files.

    CSV settings are supplied provenance, not a claim that a process completed.
    Only epoch-end accuracy records established by the configured iteration
    count are accepted. Evaluation policy describes the executable that made
    the log: patched full-test evaluation, or the original legacy50 schedule.
    No dataset, executable, or run.json is loaded.
    """
    if evaluation not in {"full", "legacy50"}:
        raise ValueError("WTA evaluation policy must be full or legacy50")
    dataset = _dataset(dataset)
    if dataset not in DEFAULTS:
        raise ValueError("Unknown WTA dataset")
    directory, suffix = DEFAULTS[dataset]
    logs = [base.local_path(path) for path in log_paths] if log_paths else [
        base.local_path(f"SLIDE/dataset/{directory}/training_wta.log")]
    configs = [base.local_path(path) for path in config_paths] if config_paths else [
        base.local_path(f"SLIDE/SLIDE/Config_{suffix}_wta.csv")]
    if len(configs) == 1:
        configs *= len(logs)
    if len(configs) != len(logs):
        raise ValueError("Provide one --wta-config for all WTA logs, or one per --wta-log in the same order")
    if len(set(logs)) != len(logs):
        raise ValueError("Duplicate WTA log path; select independent repetitions only")

    runs, hashes = [], set()
    common_parameters = None
    for log, config in zip(logs, configs):
        named_dataset = _dataset(log.parent.name)
        if named_dataset and named_dataset != dataset:
            raise ValueError(f"{base.relative(log)}: log directory dataset {named_dataset} "
                             f"conflicts with requested {dataset}")
        if not log.is_file():
            raise ValueError(f"WTA log not found: {base.relative(log)}. Collect a SLIDE run, supply "
                             "--wta-log, or select --historical-wta explicitly.")
        if not config.is_file():
            raise ValueError(f"WTA configuration not found: {base.relative(config)}")
        integers, parameters = _read_config(config, dataset)
        if common_parameters is not None and parameters != common_parameters:
            keys = sorted(key for key in parameters.keys() | common_parameters.keys()
                          if parameters.get(key) != common_parameters.get(key))
            raise ValueError(f"WTA repetitions have different configuration settings: {', '.join(keys)}")
        common_parameters = parameters
        content = log.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate WTA log content; copied logs are not independent repetitions")
        hashes.add(digest)
        rows, audit = base.parse_log(
            content.decode("utf-8-sig"),
            epoch_batches=integers["totRecords"] // integers["Batchsize"],
            configured_epochs=integers["Epoch"])
        if not rows:
            raise ValueError(f"{base.relative(log)}: no usable WTA epoch-end evaluation records")
        complete_test_batches = integers["totRecordsTest"] // integers["Batchsize"]
        for row in rows:
            full_test = evaluation == "full" or row["epoch"] == integers["Epoch"] - 1
            if not full_test and complete_test_batches < 50:
                raise ValueError(f"{base.relative(log)}: non-final WTA evaluation requests 50 test batches, "
                                 f"but the CSV provides only {complete_test_batches}")
            row["evaluation_scope"] = "all_complete_test_batches" if full_test else "first_50_test_batches"
            row["test_batches"] = complete_test_batches if full_test else 50
            row["test_records"] = row["test_batches"] * integers["Batchsize"]
        runs.append({"dataset": dataset, "run_id": base.relative(log), "repeat": None,
                     "rows": rows, "audit": audit, "parameters": parameters,
                     "config": base.relative(config), "config_sha256": base.sha256(config),
                     "log_sha256": digest, "status": "evaluation_records_only",
                     "evaluation_policy": evaluation})
    return runs


def summarize_wta(runs):
    """Aggregate independent WTA runs while retaining their evaluation scope."""
    scopes = {}
    for run in runs:
        for row in run["rows"]:
            key = (run["dataset"], row["epoch"])
            scope = {field: row[field] for field in ("evaluation_scope", "test_batches", "test_records")}
            if key in scopes and scopes[key] != scope:
                raise ValueError(f"Inconsistent WTA evaluation scope at {key}")
            scopes[key] = scope
    summary = base.aggregate(runs)
    for row in summary:
        row.update(scopes[(row["dataset"], row["epoch"])])
    return summary
