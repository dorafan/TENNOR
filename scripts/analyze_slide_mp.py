#!/usr/bin/env python3
"""Compare SLIDE-MP accuracy with fresh SLIDE WTA logs or an explicit historical reference."""

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

import analysis_common as base


# Historical WTA 50HT measurements, copied unchanged from the original manual
# plotting tables. These constants are an explicitly selected fallback; the manual
# scripts and their MP-WTA columns are not read or executed.
# Each pair is (zero-based epoch, P@1).
WTA_REFERENCE = {
    "wiki10": (  # Originally mpwta-new-wiki10.py
        (0, 0.25622),
        (1, 0.342385),
        (2, 0.425819),
        (3, 0.438107),
        (4, 0.45085),
        (5, 0.454794),
        (6, 0.512136),
        (7, 0.512591),
        (8, 0.530036),
        (9, 0.553095),
        (10, 0.519873),
        (11, 0.538987),
        (12, 0.515928),
        (13, 0.533829),
        (14, 0.537166),
        (15, 0.562955),
        (16, 0.530643),
        (17, 0.562803),
        (18, 0.549302),
        (19, 0.578125),
        (20, 0.6027),
        (21, 0.577063),
        (22, 0.563258),
        (23, 0.59375),
        (24, 0.553095),
        (25, 0.565989),
        (26, 0.583283),
        (27, 0.595419),
        (28, 0.599818),
        (29, 0.580097),
    ),
    "wiki325": (  # Originally mpwta-new-wiki325.py
        (0, 0.107983),
        (1, 0.0995455),
        (2, 0.113344),
        (3, 0.156379),
        (4, 0.185476),
        (5, 0.181391),
        (6, 0.212723),
        (7, 0.212839),
        (8, 0.263528),
        (9, 0.274334),
        (10, 0.281303),
        (11, 0.277362),
        (12, 0.322536),
        (13, 0.342423),
        (14, 0.361106),
        (15, 0.365227),
        (16, 0.373082),
        (17, 0.386657),
        (18, 0.405265),
        (19, 0.400903),
        (20, 0.422839),
        (21, 0.422396),
        (22, 0.408458),
        (23, 0.409538),
        (24, 0.420037),
        (25, 0.427652),
        (26, 0.43621),
        (27, 0.436614),
        (28, 0.437422),
        (29, 0.427265),
    ),
    "amazon670": (  # Originally mpwta-new-amazon670.py
        (0, 0.00717181),
        (1, 0.0702014),
        (2, 0.187252),
        (3, 0.223274),
        (4, 0.252916),
        (5, 0.270469),
        (6, 0.2892),
        (7, 0.2882),
        (8, 0.293495),
        (9, 0.299536),
        (10, 0.304406),
        (11, 0.306538),
        (12, 0.305485),
        (13, 0.305688),
        (14, 0.307917),
        (15, 0.30876),
        (16, 0.308617),
        (17, 0.309741),
        (18, 0.317887),
        (19, 0.311101),
        (20, 0.313677),
        (21, 0.312291),
        (22, 0.312467),
        (23, 0.311885),
        (24, 0.313088),
        (25, 0.314121),
        (26, 0.312095),
        (27, 0.311454),
        (28, 0.3152),
        (29, 0.313036),
    ),
    "amzprime": (  # Originally mpwta-new-amzprime.py
        (0, 0.00333807),
        (1, 0.0224432),
        (2, 0.0784801),
        (3, 0.119531),
        (4, 0.151847),
        (5, 0.164631),
        (6, 0.179474),
        (7, 0.174361),
        (8, 0.178622),
        (9, 0.181534),
        (10, 0.182102),
        (11, 0.186577),
        (12, 0.186648),
        (13, 0.186719),
        (14, 0.189418),
        (15, 0.189205),
        (16, 0.188849),
        (17, 0.189134),
        (18, 0.190341),
        (19, 0.189276),
        (20, 0.189347),
        (21, 0.188991),
        (22, 0.19233),
        (23, 0.190767),
        (24, 0.191264),
        (25, 0.19304),
        (26, 0.190767),
        (27, 0.191193),
        (28, 0.191974),
        (29, 0.191477),
    ),
}


def read_wta(dataset):
    """Return the embedded historical baseline and its source provenance."""
    dataset = normalize_dataset(dataset)
    source = base.relative(Path(__file__))
    rows = [{"dataset": dataset, "epoch": epoch, "p_at_1": accuracy, "source": source}
            for epoch, accuracy in WTA_REFERENCE[dataset]]
    return rows, {"path": source, "sha256": base.sha256(Path(__file__)),
                  "constant": "WTA_REFERENCE", "dataset": dataset,
                  "kind": "embedded historical WTA 50HT reference; not rerun by this analysis"}

DATASET_ALIASES = {
    "wiki10": ("wiki10", "wiki1031k", "wiki31k"),
    "wiki325": ("wiki325", "wiki325k", "wikilshtc", "wikilshtc325", "wikilshtc325k"),
    "amazon670": ("amazon670k", "amazon670", "amazon", "amz", "amz670", "amz670k"),
    "amzprime": ("amazonprime", "amazonsub", "amzprime", "amzsub"),
}
SEPARATORS = r"[\s_-]*"
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMBERS = NUMBER + r"(?:\s*,\s*" + NUMBER + r")*"
PARAMETER_ALIASES = {re.sub(r"[\s_-]", "", key).casefold(): key
                     for key in base.COMPARISON_KEYS + ["PADSIZE"]}
PARAMETER_ALIASES.update({
    "bucket": "BUCKETSIZE", "pad": "PADSIZE", "batch": "Batchsize", "bs": "Batchsize",
    "learningrate": "Lr", "threads": "threads", "thread": "threads", "th": "threads",
    "ompnumthreads": "threads", "numthreads": "threads", "threatnum": "threads",
})
IGNORED_METADATA = {"epoch", "epochinit", "loadweight", "stepsize", "islog", "ishopper",
                    "traindata", "testdata", "weight", "savedweight", "logfile", "retrivalfile"}
PATH_KEYS = {"traindata", "testdata", "weight", "savedweight", "logfile", "retrivalfile"}


def compact(value):
    return re.sub(r"[\s_-]+", "", str(value)).casefold()


def normalize_dataset(value):
    key = compact(value)
    for dataset, aliases in DATASET_ALIASES.items():
        if key in aliases:
            return dataset
    raise ValueError(f"Unknown dataset {value!r}; use wiki10, wiki325, amazon670k, or amazonsub.")


def flexible(word):
    return SEPARATORS.join(re.escape(character) for character in word)


def dataset_prefix(name):
    matches = []
    for dataset, aliases in DATASET_ALIASES.items():
        for alias in aliases:
            match = re.match(flexible(alias) + r"(?=$|[\s_-])", name, re.IGNORECASE)
            if match:
                matches.append((match.end(), dataset))
    return max(matches) if matches else (0, None)


def parameter_key(key):
    token = compact(key)
    if not re.fullmatch(r"[a-z][a-z0-9]*", token):
        raise ValueError(f"Invalid parameter name: {key!r}")
    return PARAMETER_ALIASES.get(token, token.upper())


def parameter_value(raw):
    parts = str(raw).strip().split(",")
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            raise ValueError(f"Empty parameter value: {raw!r}")
        if re.fullmatch(r"[+-]?" + NUMBER, part):
            number = Decimal(part)
            # Limit pathological exponents before formatting a decimal string.
            if abs(number.adjusted()) > 100:
                raise ValueError(f"Parameter exponent is too large: {raw!r}")
            result.append("0" if not number else format(number.normalize(), "f"))
        else:
            result.append(part.casefold())
    return ",".join(result)


def add_parameter(parameters, key, raw, source):
    if compact(key) in PATH_KEYS:
        raise ValueError(f"{source}: {key} is an input/output path, not a configuration filter")
    key, value = parameter_key(key), parameter_value(raw)
    if key in parameters and parameters[key] != value:
        raise ValueError(f"{source}: conflicting {key}: {parameters[key]} vs {value}")
    parameters[key] = value


def parse_run_name(name):
    end, dataset = dataset_prefix(name.strip())
    result = {"dataset": dataset, "parameters": {}, "repeat": None}
    if dataset is None:
        return result  # Legacy opaque names can still be identified by run.json.
    rest = name.strip()[end:]
    while rest.strip(" _-\t"):
        rest = rest.lstrip(" _-\t")
        method = re.match(r"mp(?:[\s_-]*wta)?(?=$|[\s_-])", rest, re.IGNORECASE)
        if method:
            rest = rest[method.end():]
            continue
        matched = False
        for alias in sorted(PARAMETER_ALIASES, key=len, reverse=True):
            expression = (flexible(alias) + r"(?:[\s_]*=[\s_]*|[\s_-]*)"
                          + "(" + NUMBERS + r")(?=$|[\s_-])")
            match = re.match(expression, rest, re.IGNORECASE)
            if match:
                add_parameter(result["parameters"], alias, match[1], name)
                rest = rest[match.end():]
                matched = True
                break
        if matched:
            continue
        repeat = re.fullmatch(r"(?:(?:repeat|rep|run|trial)[\s_=-]*)?(\d+)", rest, re.IGNORECASE)
        if repeat:
            result["repeat"] = str(int(repeat[1]))
            break
        # Unknown named numeric parameters remain configuration dimensions too.
        generic = re.match(r"([a-z][a-z0-9_]*)\s*=\s*(" + NUMBERS
                           + r"|[a-z][a-z0-9.]*)(?=$|[\s_-])", rest, re.IGNORECASE)
        if generic:
            add_parameter(result["parameters"], generic[1], generic[2], name)
            rest = rest[generic.end():]
            continue
        raise ValueError(f"{name}: cannot interpret {rest!r}. Use named parameters, e.g. "
                         "wiki10-bucketsize128-batchsize32-1; the final number is the repeat ID.")
    return result


def metadata_dataset(metadata):
    values = metadata.get("effective_values", {})
    evidence = []
    paths = [values.get("trainData", ""), values.get("testData", "")]
    paths += [item.get("path", "") for item in metadata.get("inputs", []) if isinstance(item, dict)]
    for raw in paths:
        try:
            evidence.append(normalize_dataset(base.portable_path(raw).parent.name))
        except ValueError:
            pass
    config = base.portable_path(metadata.get("config", "")).stem
    config = re.sub(r"^config[\s_-]+", "", config, flags=re.IGNORECASE)
    _, dataset = dataset_prefix(config)
    if dataset:
        evidence.append(dataset)
    if len(set(evidence)) > 1:
        raise ValueError(f"run.json contains conflicting dataset identities: {sorted(set(evidence))}")
    return evidence[0] if evidence else None


def metadata_parameters(metadata, include_ignored=False):
    parameters = {}
    for key, value in metadata.get("effective_values", {}).items():
        if compact(key) not in (PATH_KEYS if include_ignored else IGNORED_METADATA):
            add_parameter(parameters, key, value, "run.json")
    threads = metadata.get("openmp", {}).get("OMP_NUM_THREADS")
    if threads is not None:
        add_parameter(parameters, "threads", threads, "run.json")
    return parameters


def discover_runs(runs_dir, dataset, include_names=None, parameter_keys=()):
    dataset = normalize_dataset(dataset)
    runs_dir = base.local_path(runs_dir)
    if not runs_dir.is_dir():
        raise ValueError(f"Runs directory does not exist: {runs_dir}")
    runs, skipped = [], []
    for directory in sorted(runs_dir.iterdir()):
        if not directory.is_dir():
            continue
        if include_names and directory.name not in include_names:
            continue
        directory = base.local_path(directory)
        _, named_dataset = dataset_prefix(directory.name)
        metadata, manifest = base.read_metadata(directory)
        recorded_dataset = metadata_dataset(metadata)
        if named_dataset and recorded_dataset and named_dataset != recorded_dataset:
            raise ValueError(f"{directory.name}: folder dataset {named_dataset} conflicts with "
                             f"run.json dataset {recorded_dataset}")
        actual_dataset = named_dataset or recorded_dataset
        if actual_dataset != dataset:
            if actual_dataset is None:
                skipped.append({"input": base.relative(directory), "reason": "dataset not recognized"})
            continue
        named = parse_run_name(directory.name)
        parameters = metadata_parameters(metadata)
        recorded = metadata_parameters(metadata, include_ignored=True)
        for key, value in named["parameters"].items():
            if key in recorded and recorded[key] != value:
                raise ValueError(f"{directory.name}: folder vs run.json: conflicting {key}: "
                                 f"{value} vs {recorded[key]}")
            add_parameter(parameters, key, value, f"{directory.name}: folder vs run.json")
        if parameters.get("isMPWTA", "1") != "1" or parameters.get("isBruteForce", "0") != "0":
            skipped.append({"input": base.relative(directory), "reason": "not a sparse MP-WTA run"})
            continue
        try:
            log = base.locate_log(directory, metadata)
        except FileNotFoundError as error:
            skipped.append({"input": base.relative(directory), "reason": str(error)})
            continue
        content = log.read_bytes()
        rows, audit = base.parse_log(content.decode("utf-8-sig"), base.tail_epoch(metadata))
        if not rows:
            skipped.append({"input": base.relative(directory), "reason": "no complete epochs", "audit": audit})
            continue
        runs.append({"dataset": dataset, "run_id": base.relative(log), "folder": directory.name,
                     "runs_directory": base.relative(runs_dir),
                     "parameters": parameters, "folder_parameters": named["parameters"],
                     "recorded_parameters": recorded,
                     "repeat": named["repeat"], "rows": rows, "audit": audit,
                     "log_sha256": hashlib.sha256(content).hexdigest(),
                     "metadata": base.relative(manifest) if manifest else None,
                     "metadata_sha256": base.sha256(manifest) if manifest else None,
                     "status": metadata.get("status", "unknown"),
                     "confirmed_terminal_epoch": base.tail_epoch(metadata)})
    # Epoch limits and logging settings do not split runs by default. If a user
    # explicitly names/filters one, include its recorded value for every run.
    explicit_keys = {parameter_key(key) for key in parameter_keys}
    explicit_keys.update(key for run in runs for key in run["folder_parameters"])
    for run in runs:
        for key in explicit_keys:
            if key in run["recorded_parameters"]:
                run["parameters"][key] = run["recorded_parameters"][key]
    return runs, skipped


def configuration_groups(runs):
    keys = sorted({key for run in runs for key in run["parameters"]})
    groups = {}
    for run in runs:
        signature = tuple((key, run["parameters"].get(key)) for key in keys)
        groups.setdefault(signature, []).append(run)
    return keys, groups


def path_from_cwd(path):
    """Display a usable path from the directory where the reviewer invoked us."""
    try:
        return os.path.relpath(path, Path.cwd()).replace("\\", "/")
    except ValueError:  # Different Windows drives cannot have a relative path.
        return Path(path).as_posix()


def invocation_command():
    # Preserve the active interpreter, including a shared virtual environment.
    # Do not resolve its symlinks: that could silently leave the environment.
    interpreter = Path(sys.executable).absolute()
    python = path_from_cwd(interpreter) if base.ROOT in interpreter.parents else interpreter.as_posix()
    return [python, path_from_cwd(Path(__file__).resolve())]


def describe_choices(runs, filters=None):
    keys, groups = configuration_groups(runs)
    differing = [key for key in keys if len({run["parameters"].get(key) for run in runs}) > 1]
    shown = differing or keys
    lines = ["Configurations found (missing values are UNKNOWN):"]
    for index, (signature, members) in enumerate(groups.items(), 1):
        values = dict(signature)
        details = ", ".join(f"{key}={values[key] if values[key] is not None else 'UNKNOWN'}" for key in shown)
        lines.append(f"  [{index}] {details or '(no recorded parameters)'}; runs={len(members)}")
        for run in members:
            lines.append(f"      {run['run_id']}")
        command = invocation_command() + ["--dataset", members[0]["dataset"]]
        runs_directory = members[0].get("runs_directory", "SLIDE-MP/dataset/runs")
        if runs_directory != "SLIDE-MP/dataset/runs":
            command += ["--runs-dir", runs_directory]
        command_filters = dict(filters or {})
        if differing and all(values[key] is not None for key in differing):
            command_filters.update({key: values[key] for key in differing})
        else:
            for run in members:
                command += ["--run", run.get("folder", Path(run["run_id"]).parent.name)]
        for key, value in sorted(command_filters.items()):
            option = {"BUCKETSIZE": "--bucketsize", "PADSIZE": "--padsize",
                      "Batchsize": "--batchsize"}.get(key)
            command += [option, value] if option else ["--param", f"{key}={value}"]
        lines.append("      Select: " + ("& " if os.name == "nt" else "") + shlex.join(command))
    return "\n".join(lines)


class AmbiguousSelection(ValueError):
    pass


def select_runs(runs, filters):
    normalized = {}
    for key, value in filters.items():
        add_parameter(normalized, key, value, "filters")
    selected = [run for run in runs if all(run["parameters"].get(key) == value
                                          for key, value in normalized.items())]
    if not selected:
        raise ValueError("No runs match the requested parameters.\n" + describe_choices(runs))
    _, groups = configuration_groups(selected)
    if len(groups) > 1:
        raise AmbiguousSelection("Multiple configurations remain; provide the distinguishing parameters "
                                 "or --run names before averaging.\n" + describe_choices(selected, normalized))
    identities = set()
    for run in selected:
        repeat = run.get("repeat")
        if repeat is not None:
            identity = (run["dataset"], repeat)
            if identity in identities:
                raise ValueError(f"Duplicate repeat ID {repeat} for the selected configuration. "
                                 "Use --run to select independent experiments; copied logs must not be averaged twice.")
            identities.add(identity)
    return selected


def summarize_mp(runs):
    """Attach the full-test evaluation size when the recorded settings establish it."""
    summary = base.aggregate(runs)
    sizes = {}
    for run in runs:
        values = run["parameters"]
        batch, count = values.get("Batchsize", ""), values.get("totRecordsTest", "")
        if (re.fullmatch(r"[1-9][0-9]*", str(batch))
                and re.fullmatch(r"[1-9][0-9]*", str(count)) and int(count) >= int(batch)):
            sizes[run["run_id"]] = (int(count) // int(batch), int(batch))
        else:
            sizes[run["run_id"]] = None
    for row in summary:
        contributing = {sizes[run_id] for run_id in row["run_ids"]}
        size = next(iter(contributing)) if len(contributing) == 1 else None
        row.update(evaluation_scope="all_complete_test_batches",
                   test_batches=size[0] if size else None,
                   test_records=size[0] * size[1] if size else None)
    return summary


def compare_accuracy(summary, wta_summary, reference):
    """Keep both curves; compute a difference only for known matching test prefixes."""
    measured = {row["epoch"]: row for row in summary}
    baseline = {row["epoch"]: row for row in (wta_summary or reference)}
    kind = "measured_logs" if wta_summary else "historical"
    rows = []
    for epoch in sorted(measured.keys() | baseline.keys()):
        mp, wta = measured.get(epoch, {}), baseline.get(epoch, {})
        mp_mean = mp.get("p_at_1_mean")
        wta_mean = wta.get("p_at_1_mean", wta.get("p_at_1"))
        mp_count, wta_count = mp.get("test_records"), wta.get("test_records")
        match = mp_count == wta_count if mp_count is not None and wta_count is not None else None
        rows.append({"dataset": summary[0]["dataset"], "epoch": epoch,
                     "baseline_kind": kind,
                     "mp_p_at_1_mean": mp_mean, "wta_p_at_1_mean": wta_mean,
                     "mp_n_runs": mp.get("n_runs", 0),
                     "wta_n_runs": wta.get("n_runs", 0) if kind == "measured_logs" else None,
                     "mp_test_records": mp_count, "wta_test_records": wta_count,
                     "mp_evaluation_scope": mp.get("evaluation_scope"),
                     "wta_evaluation_scope": wta.get("evaluation_scope", "unknown") if wta else None,
                     "evaluation_match": match,
                     "p_at_1_difference": mp_mean - wta_mean if match is True else None})
    return rows


def write_results(runs, skipped, dataset, filters, output_arg, command, *,
                  wta_runs=None, historical=False):
    summary = summarize_mp(runs)
    if historical:
        reference, reference_source = read_wta(dataset)
        wta_runs, wta_summary = [], []
    else:
        import wta_logs
        if not wta_runs:
            raise ValueError("Select WTA log inputs or --historical-wta explicitly")
        wta_summary = wta_logs.summarize_wta(wta_runs)
        reference = []
        reference_source = {"kind": "measured_logs", "dataset": dataset,
                            "run_ids": [run["run_id"] for run in wta_runs],
                            "evaluation_policies": sorted({run["evaluation_policy"] for run in wta_runs}),
                            "loader_sha256": base.sha256(Path(wta_logs.__file__))}
    comparison = compare_accuracy(summary, wta_summary, reference)
    parameters = runs[0]["parameters"]
    output = base.local_path(output_arg or f"analysis/slide-mp/{dataset}")
    # Also protect explicitly selected, archived inputs outside component trees.
    destinations = {base.local_path(output / name) for name in (
        "epochs.csv", "summary.csv", "wta_epochs.csv", "wta_summary.csv",
        "wta_reference.csv", "comparison.csv", "analysis.json",
        f"{dataset}_epoch_accuracy.pdf", f"{dataset}_epoch_accuracy.png")}
    for protected in ("SLIDE", "SLIDE-MP", "SLIDE-ORAM", "TENNOR", "scripts"):
        path = base.local_path(base.ROOT / protected)
        for destination in destinations | {output, base.local_path(output / ".matplotlib")}:
            if destination == path or path in destination.parents:
                raise ValueError("Use an analysis output directory separate from inputs and scripts.")
    for run in runs + wta_runs:
        for key in ("run_id", "config", "metadata"):
            if run.get(key) and base.local_path(run[key]) in destinations:
                raise ValueError("An analysis output would overwrite a selected input")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(base.local_path(output / ".matplotlib"))
    details = [dict(row, dataset=dataset, run_id=run["run_id"], repeat=run["repeat"])
               for run in runs for row in run["rows"]]
    base.write_csv(output / "epochs.csv", ["dataset", "run_id", "repeat", "epoch", "batches", "training_time",
                                           "p_at_1", "header_line", "metric_line", "occurrence"], details)
    summary_columns = ["dataset", "epoch", "p_at_1_mean", "p_at_1_std", "n_runs", "run_ids",
                       "evaluation_scope", "test_batches", "test_records"]
    base.write_csv(output / "summary.csv", summary_columns, summary)
    wta_details = [dict(row, dataset=dataset, run_id=run["run_id"], config=run["config"])
                   for run in wta_runs for row in run["rows"]]
    base.write_csv(output / "wta_epochs.csv", ["dataset", "run_id", "config", "epoch", "batches", "training_time",
                 "p_at_1", "header_line", "metric_line", "occurrence", "evaluation_scope",
                 "test_batches", "test_records"], wta_details)
    base.write_csv(output / "wta_summary.csv", summary_columns, wta_summary)
    base.write_csv(output / "comparison.csv", ["dataset", "epoch", "baseline_kind", "mp_p_at_1_mean",
                 "wta_p_at_1_mean", "mp_n_runs", "wta_n_runs", "mp_test_records", "wta_test_records",
                 "mp_evaluation_scope", "wta_evaluation_scope", "evaluation_match", "p_at_1_difference"], comparison)
    base.write_csv(output / "wta_reference.csv", ["dataset", "epoch", "p_at_1", "source"], reference)
    version = base.make_plot(dataset, summary, reference, output, wta_summary=wta_summary)
    report = {"schema_version": 4, "command": command, "dataset": dataset, "filters": filters,
              "parameters": parameters, "python": sys.version,
              "matplotlib": version, "analyzer_sha256": base.sha256(Path(__file__)),
              "core_sha256": base.sha256(Path(base.__file__)), "reference": reference_source,
              "rules": {"selection": "last evaluation in latest complete epoch occurrence per log",
                        "completion": "later header or matched successful run.json at configured final epoch",
                        "wta_completion": "CSV-derived exact epoch-end iteration confirms evaluation, not process exit",
                        "wta_evaluation": "full: all complete test batches at every epoch end; explicit legacy50: first 50 test batches before final epoch, all complete test batches at final epoch; policy supplied by user",
                        "comparison": "MP-WTA minus WTA only when both declared test-prefix sizes are known and equal; dataset contents are not revalidated",
                        "mean": "equal weight per log at each epoch; no filling missing values",
                        "spread": "sample SD (ddof=1), absent when n=1",
                        "missing_parameters": "UNKNOWN is distinct from every known value"},
              "runs": runs, "wta_runs": wta_runs, "comparison": comparison,
              "skipped_inputs": skipped}
    base.local_path(output / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    counts = [row["n_runs"] for row in summary]
    print(f"{dataset}: {len(runs)} independent runs, {len(summary)} epochs, "
          f"samples per epoch {min(counts)}..{max(counts)}")
    if wta_runs:
        print(f"WTA: {len(wta_runs)} log(s), {len(wta_summary)} epochs; source: measured_logs")
        if any(row["evaluation_match"] is not True for row in comparison):
            print("Some epochs have different or unknown evaluation sizes, or missing measurements; "
                  "their accuracy differences are left blank. See comparison.csv.")
    else:
        print("WTA source: embedded historical reference (--historical-wta); no WTA logs read")
    print(f"Outputs: {path_from_cwd(output)}/")
    if output_arg is None:
        print(f"For custom experiment parameters, use a named output subfolder, e.g. "
              f"--output-dir analysis/slide-mp/{dataset}/custom-name")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="case/separator-insensitive dataset name or alias")
    parser.add_argument("--runs-dir", default="SLIDE-MP/dataset/runs", help="directory inside the artifact")
    parser.add_argument("--run", action="append", default=[], help="exact folder name to include; repeat for multiple runs")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--wta-log", action="append", default=[], metavar="PATH",
                          help="WTA training log, relative to artifact root; repeat for independent runs; defaults to SLIDE/dataset/<dataset>/training_wta.log")
    baseline.add_argument("--historical-wta", action="store_true",
                          help="explicitly use the embedded historical WTA curve instead of new logs")
    parser.add_argument("--wta-config", action="append", default=[], metavar="PATH",
                        help="actual WTA CSV: one shared config or one per --wta-log; defaults to the dataset's Config_*_wta.csv")
    parser.add_argument("--wta-evaluation", choices=("full", "legacy50"),
                        help="WTA executable's epoch-end evaluation policy: full (default for this artifact), or legacy50 for logs from unmodified upstream SLIDE")
    parser.add_argument("--bucketsize", "--bucket-size", "--bucket_size", action="append", default=[])
    parser.add_argument("--padsize", "--pad-size", "--pad_size", action="append", default=[])
    parser.add_argument("--batchsize", "--batch-size", "--batch_size", action="append", default=[])
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="additional configuration filter")
    parser.add_argument("--list", action="store_true", help="list discovered configurations without writing outputs")
    parser.add_argument("--output-dir", help="relative to artifact root; use a named dataset subfolder for custom parameters")
    args = parser.parse_args(argv)
    if args.historical_wta and (args.wta_config or args.wta_evaluation):
        raise ValueError("--wta-config/--wta-evaluation cannot be combined with --historical-wta")
    dataset = normalize_dataset(args.dataset)
    filters = {}
    for key, values in (("BUCKETSIZE", args.bucketsize), ("PADSIZE", args.padsize), ("Batchsize", args.batchsize)):
        for value in values:
            normalized = parameter_value(value)
            if not re.fullmatch(r"[1-9][0-9]*", normalized):
                raise ValueError(f"{key} must be a positive integer")
            add_parameter(filters, key, normalized, "command line")
    for item in args.param:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"--param expects KEY=VALUE, got {item!r}")
        add_parameter(filters, key, value, "command line")
    runs, skipped = discover_runs(args.runs_dir, dataset, include_names=args.run, parameter_keys=filters)
    for item in skipped:
        print(f"Skipped {item['input']}: {item['reason']}")
    if args.run:
        missing = set(args.run) - {run["folder"] for run in runs}
        if missing:
            raise ValueError(f"Requested folders have no usable {dataset} runs: {sorted(missing)}")
        runs = [run for run in runs if run["folder"] in args.run]
    if not runs:
        raise ValueError(f"No usable {dataset} experiments found in {args.runs_dir}.")
    if args.list:
        matches = [run for run in runs if all(run["parameters"].get(k) == v for k, v in filters.items())]
        if not matches:
            raise ValueError("No runs match the requested parameters.\n" + describe_choices(runs))
        print(describe_choices(matches, filters))
        return 0
    selected = select_runs(runs, filters)
    print(f"Selected {len(selected)} runs for {dataset}; parameters: "
          + json.dumps(selected[0]["parameters"], sort_keys=True))
    wta_runs = None
    if not args.historical_wta:
        import wta_logs
        wta_runs = wta_logs.load_wta_runs(args.wta_log, args.wta_config, dataset,
                                       evaluation=args.wta_evaluation or "full")
        mp_hashes = {run["log_sha256"] for run in selected}
        if any(run["log_sha256"] in mp_hashes for run in wta_runs):
            raise ValueError("The WTA and MP-WTA selections contain the same log content; use independent method outputs")
    write_results(selected, skipped, dataset, filters, args.output_dir,
                  sys.argv if argv is None else argv, wta_runs=wta_runs, historical=args.historical_wta)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, SyntaxError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
