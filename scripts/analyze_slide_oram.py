#!/usr/bin/env python3
"""Analyze SLIDE-ORAM console logs and save one Table 2 row, without plotting."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import sys

import analysis_common as base
import analyze_slide_mp as naming
from oram_log import parse_console


# The artifact's SLIDE-ORAM implementation is confirmed to use one thread.
DEFAULT_TABLE_THREADS = 1


def positive_integer(parameters, key):
    value = str(parameters.get(key, ""))
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"A positive {key} is required in the recorded configuration.")
    return int(value)


def summarize_runs(runs, parameters, skip_batches=0, max_batches=None, rehash_seconds=None):
    batchsize = positive_integer(parameters, "Batchsize")
    records = positive_integer(parameters, "totRecords")
    rehash = positive_integer(parameters, "Rehash")
    if rehash < batchsize or rehash % batchsize:
        raise ValueError("Rehash must be a positive multiple of Batchsize.")
    if records < batchsize:
        raise ValueError("totRecords must contain at least one complete batch.")
    if skip_batches < 0 or (max_batches is not None and max_batches < 1):
        raise ValueError("skip_batches must be nonnegative; max_batches must be positive.")
    if rehash_seconds is not None and (not math.isfinite(rehash_seconds) or rehash_seconds < 0):
        raise ValueError("--rehash-seconds must be finite and nonnegative.")
    windows, excluded_runs = [], []
    for run in runs:
        window = run["batches"][skip_batches:None if max_batches is None else skip_batches + max_batches]
        if window:
            windows.append((run, window))
        else:
            excluded_runs.append(run["run_id"])
    if not windows:
        raise ValueError("No complete batches remain in the requested window.")

    events, rejected_events = [], []
    for run, _ in windows:
        for batch in run["batches"]:
            if not batch.get("rehash_present"):
                continue
            total, seconds = batch.get("reported_seconds"), batch.get("rehash_s")
            event = {"run_id": run["run_id"], "batch": batch["batch"],
                     "header_line": batch.get("header_line"), "seconds": seconds}
            if (not batch.get("rehash_valid") or total is None or seconds is None
                    or not math.isfinite(total) or not math.isfinite(seconds)
                    or seconds < 0 or total < seconds):
                rejected_events.append(dict(event, reason="invalid rehash timing or rehash exceeds batch total"))
            else:
                events.append(event)
    if rehash_seconds is None and not events:
        raise ValueError("No complete, timed rehash event is available for amortization. "
                         "Include a run containing one, or supply a measured --rehash-seconds value.")
    event_mean = rehash_seconds if rehash_seconds is not None else statistics.mean(e["seconds"] for e in events)
    # Refresh cadence is fixed by the configuration, never by the number of
    # observed batches or the length of the selected measurement window.
    interval = rehash // batchsize
    overhead = event_mean / interval
    per_run = []
    for run, window in windows:
        for batch in window:
            total, seconds = batch.get("reported_seconds"), batch.get("rehash_s")
            if total is None or not math.isfinite(total) or total < 0:
                raise ValueError(f"{run['run_id']}, batch {batch['batch']}: invalid batch duration.")
            if (not batch.get("rehash_valid", True) or seconds is None
                    or not math.isfinite(seconds) or seconds < 0):
                raise ValueError(f"{run['run_id']}, batch {batch['batch']}: invalid or incomplete rehash timing.")
            if seconds > total:
                raise ValueError(f"{run['run_id']}, batch {batch['batch']}: rehash exceeds batch time.")
        raw = statistics.mean(b["reported_seconds"] for b in window)
        core = statistics.mean(b["reported_seconds"] - b["rehash_s"] for b in window)
        per_run.append({"run_id": run["run_id"], "n_batches": len(window),
                        "selected_batches": [{"batch": b["batch"], "header_line": b.get("header_line")} for b in window],
                        "raw_batch_seconds": raw, "core_batch_seconds": core, "batch_seconds": core + overhead,
                        "batch_hours": (core + overhead) / 3600})
    core = statistics.mean(run["core_batch_seconds"] for run in per_run)
    batch_seconds = core + overhead
    notes = ["Batch time uses adjacent consecutive START TRAINING batch timestamps within one execution.",
             "The final batch without a following batch-start boundary is excluded, even after a successful run.",
             "Launcher timestamps have one-second resolution and reflect line receipt, not internal high-resolution timers.",
             "Rehash calibration uses all valid events in complete batches of contributing logs, including outside the selected window.",
             "Rehash is amortized over Rehash / Batchsize configured batches, independent of the observed or selected batch count.",
             "Network initialization and restart downtime are excluded; batch 0 is not automatically discarded.",
             "Epoch time is an extrapolation using floor(totRecords / Batchsize)."]
    if rejected_events:
        notes.append("Invalid rehash calibration candidates were excluded; see rejected_rehash_events.")
    if excluded_runs:
        notes.append("Runs with no complete batches in the selected window do not contribute.")
    return {"batch_seconds": batch_seconds, "batch_hours": batch_seconds / 3600,
            "raw_batch_seconds": statistics.mean(run["raw_batch_seconds"] for run in per_run),
            "core_batch_seconds": core, "rehash_amortized_seconds": overhead,
            "rehash_event_mean_seconds": event_mean, "rehash_event_count": len(events),
            "rehash_interval_batches": interval, "rehash_events": events,
            "rejected_rehash_events": rejected_events,
            "rehash_calibration_source": "explicit_override" if rehash_seconds is not None else "complete_batches_in_contributing_logs",
            "n_runs": len(per_run), "n_batches": sum(run["n_batches"] for run in per_run),
            "n_inputs": sum(run["n_batches"] for run in per_run) * batchsize,
            "batches_per_epoch": records // batchsize, "unused_epoch_records": records % batchsize,
            "epoch_hours": batch_seconds * (records // batchsize) / 3600,
            "run_mean_std_seconds": statistics.stdev(run["batch_seconds"] for run in per_run) if len(per_run) > 1 else None,
            "run_summaries": per_run, "excluded_runs": excluded_runs,
            "window": {"skip_batches": skip_batches, "max_batches": max_batches},
            "averaging": "equal weight per independent run after averaging its selected complete batches",
            "notes": notes}


def case_file(directory, filename):
    matches = [p for p in directory.iterdir() if p.is_file() and p.name.casefold() == filename.casefold()]
    if len(matches) > 1:
        raise ValueError(f"{directory}: ambiguous filename variants for {filename}")
    return base.local_path(matches[0]) if matches else None


def read_config(path):
    values = {}
    for number, raw in enumerate(base.local_path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        if key in values and values[key] != value:
            raise ValueError(f"{path}:{number}: conflicting {key}")
        values[key] = value
    return values


def parse_run_name(name):
    end, dataset = naming.dataset_prefix(name.strip())
    if dataset is None:
        return naming.parse_run_name(name)
    # Existing measurements use wiki10_oram_time. Treat only the method label
    # as decoration; unknown numeric parameter layouts still require names.
    remainder = name.strip()[end:].lstrip(" _-\t")
    method = re.match(r"(?:slide[\s_-]*)?oram(?:[\s_-]*time)?(?=$|[\s_-])", remainder, re.IGNORECASE)
    if method:
        remainder = remainder[method.end():]
    return naming.parse_run_name(dataset + ("-" + remainder if remainder else ""))


def discover_runs(runs_dir, dataset, include_names=(), config_path=None, parameter_keys=()):
    dataset = naming.normalize_dataset(dataset)
    directory = base.local_path(runs_dir)
    if not directory.is_dir():
        raise ValueError(f"Missing runs directory: {directory}")
    runs, skipped = [], []
    for run_dir in sorted(directory.iterdir()):
        if not run_dir.is_dir() or (include_names and run_dir.name not in include_names):
            continue
        run_dir = base.local_path(run_dir)
        metadata, manifest = base.read_metadata(run_dir)
        original_metadata = metadata
        config = case_file(run_dir, "config.csv") or (base.local_path(config_path) if config_path else None)
        config_used = None
        if not metadata.get("effective_values") and config is not None:
            metadata = dict(metadata, effective_values=read_config(config))
            config_used = config
        recorded_dataset = naming.metadata_dataset(metadata)
        _, named_dataset = naming.dataset_prefix(run_dir.name)
        if named_dataset and recorded_dataset and named_dataset != recorded_dataset:
            raise ValueError(f"{run_dir.name}: folder dataset conflicts with run.json/config dataset")
        if (named_dataset or recorded_dataset) != dataset:
            if not named_dataset and not recorded_dataset:
                skipped.append({"input": base.relative(run_dir), "reason": "dataset not recognized"})
            continue
        parameters = naming.metadata_parameters(metadata)
        recorded = naming.metadata_parameters(metadata, include_ignored=True)
        named = parse_run_name(run_dir.name)
        for key, value in named["parameters"].items():
            if key in recorded and recorded[key] != value:
                raise ValueError(f"{run_dir.name}: folder {key}={value} conflicts with recorded {recorded[key]}")
            naming.add_parameter(parameters, key, value, run_dir.name)
        log = case_file(run_dir, "console.log")
        if log is None:
            skipped.append({"input": base.relative(run_dir), "reason": "no console.log"})
            continue
        batchsize = positive_integer(parameters, "Batchsize")
        raw = log.read_bytes()
        parsed = parse_console(raw.decode("utf-8-sig"), batchsize)
        if not parsed["batches"]:
            skipped.append({"input": base.relative(run_dir), "reason": "no complete batches", "audit": parsed["audit"]})
            continue
        runs.append({"run_id": base.relative(log), "folder": run_dir.name, "dataset": dataset,
                     "parameters": parameters, "recorded_parameters": recorded,
                     "folder_parameters": named["parameters"], "repeat": named["repeat"],
                     "runs_directory": base.relative(directory), "log_sha256": hashlib.sha256(raw).hexdigest(),
                     "metadata": base.relative(manifest) if manifest else None,
                     "metadata_sha256": base.sha256(manifest) if manifest else None,
                     "status": original_metadata.get("status", "unknown"),
                     "openmp": original_metadata.get("openmp", {}),
                     "console_logging": original_metadata.get("console_logging", {}),
                     "config_fallback": base.relative(config_used) if config_used else None,
                     "config_sha256": base.sha256(config_used) if config_used else None,
                     **parsed})
    explicit = {naming.parameter_key(key) for key in parameter_keys}
    explicit.update(key for run in runs for key in run["folder_parameters"])
    for run in runs:
        for key in explicit:
            if key in run["recorded_parameters"]:
                run["parameters"][key] = run["recorded_parameters"][key]
    return runs, skipped


def choices(runs, filters=None):
    keys, groups = naming.configuration_groups(runs)
    differing = [key for key in keys if len({run["parameters"].get(key) for run in runs}) > 1]
    lines = ["SLIDE-ORAM configurations:"]
    for index, (signature, members) in enumerate(groups.items(), 1):
        values = dict(signature)
        shown = differing or [key for key in ("threads", "Batchsize", "Rehash") if key in values]
        detail = ", ".join(f"{key}={values.get(key) if values.get(key) is not None else 'UNKNOWN'}" for key in shown)
        lines.append(f"  [{index}] {detail}; runs={len(members)}; complete batches={sum(len(r['batches']) for r in members)}")
        for run in members:
            lines.append("      " + run["run_id"])
        interpreter = Path(sys.executable).absolute()
        executable = naming.path_from_cwd(interpreter) if base.ROOT in interpreter.parents else interpreter.as_posix()
        command = [executable, naming.path_from_cwd(Path(__file__).resolve()), "--dataset", members[0]["dataset"]]
        if members[0]["runs_directory"] != "SLIDE-ORAM/dataset/runs":
            command += ["--runs-dir", members[0]["runs_directory"]]
        selected_filters = dict(filters or {})
        if differing and all(values[key] is not None for key in differing):
            selected_filters.update({key: values[key] for key in differing})
        else:
            for run in members:
                command += ["--run", run["folder"]]
        for key, value in selected_filters.items():
            flag = {"threads": "--threads", "Batchsize": "--batchsize", "BUCKETSIZE": "--bucketsize", "PADSIZE": "--padsize"}.get(key)
            command += [flag, value] if flag else ["--param", f"{key}={value}"]
        lines.append("      Select: " + ("& " if os.name == "nt" else "") + shlex.join(command))
    return "\n".join(lines)


def select_runs(runs, filters):
    selected = [run for run in runs if all(run["parameters"].get(key) == value for key, value in filters.items())]
    if not selected:
        raise ValueError("No configuration matches.\n" + choices(runs))
    if len(naming.configuration_groups(selected)[1]) > 1:
        raise ValueError("Select one configuration before averaging; different batch sizes or other settings "
                         "are never averaged together.\n" + choices(selected, filters))
    repeats = [run["repeat"] for run in selected if run["repeat"] is not None]
    if len(repeats) != len(set(repeats)):
        raise ValueError("Duplicate repeat IDs in one configuration; select independent runs with --run.")
    return selected


def write_results(runs, skipped, summary, dataset, output_arg, command, table_threads=DEFAULT_TABLE_THREADS):
    parameters = runs[0]["parameters"]
    batchsize = int(parameters["Batchsize"])
    thread_label = str(table_threads) if table_threads is not None else "unknown"
    output = base.local_path(output_arg or f"analysis/slide-oram/{dataset}/bs{batchsize}/threads{thread_label}")
    for protected in ("TENNOR", "SLIDE-MP", "SLIDE-ORAM", "scripts"):
        path = base.ROOT / protected
        if output == path or path in output.parents:
            raise ValueError("Choose an analysis output directory outside components and scripts.")
    output.mkdir(parents=True, exist_ok=True)
    row = {"dataset": dataset, "method": "SLIDE-ORAM", "threads": table_threads, "batchsize": batchsize,
           "configured_omp_threads": parameters.get("threads"),
           "batch_hours": summary["batch_hours"], "speedup": 1.0, "epoch_hours": summary["epoch_hours"],
           "batch_seconds": summary["batch_seconds"],
           "n_runs": summary["n_runs"], "n_batches": summary["n_batches"], "n_inputs": summary["n_inputs"],
           "batches_per_epoch": summary["batches_per_epoch"], "raw_batch_seconds": summary["raw_batch_seconds"],
           "rehash_amortized_seconds": summary["rehash_amortized_seconds"],
           "timing_basis": "consecutive_batch_start_timestamps_with_rehash_amortization"}
    base.write_csv(output / "table_row.csv", list(row), [row])
    markdown = ("| Dataset | Method | #Thd | Batch(h) | SU | Ep.(h) |\n"
                "| --- | --- | ---: | ---: | ---: | ---: |\n"
                f"| {base.DATASETS[dataset][0]} | SLIDE-ORAM | {thread_label} | {row['batch_hours']:.6f} | 1.000 | {row['epoch_hours']:.3f} |\n\n"
                f"Batch size: {batchsize}. {row['n_runs']} run(s), {row['n_batches']} complete batches "
                f"({row['n_inputs']} observed inputs). Epoch time is extrapolated using {row['batches_per_epoch']} batches.\n\n"
                "SU=1 identifies this measurement as the SLIDE-ORAM baseline. Match dataset, batch size, and compatible settings before computing TENNOR speedup.\n\n"
                "Requested OMP_NUM_THREADS is recorded separately and does not by itself prove the effective parallelism.\n")
    base.local_path(output / "table_row.md").write_text(markdown, encoding="utf-8")
    selected = {(run["run_id"], batch["batch"], batch["header_line"])
                for run in summary["run_summaries"] for batch in run["selected_batches"]}
    details = []
    for run in runs:
        for batch in run["batches"]:
            identity = (run["run_id"], batch["batch"], batch.get("header_line"))
            details.append(dict(batch, run_id=run["run_id"], selected=identity in selected))
    columns = ["run_id", "batch", "segment", "occurrence", "header_line", "end_line", "selected",
               "start_time", "end_time", "reported_seconds", "rehash_present", "rehash_valid", "rehash_s",
               "update_s", "ff_count", "bp_count", "free_count", "errors"]
    base.write_csv(output / "batches.csv", columns, details)
    base.write_csv(output / "run_summary.csv", ["run_id", "n_batches", "batch_hours", "raw_batch_seconds", "core_batch_seconds", "batch_seconds"], summary["run_summaries"])
    result = {"schema_version": 1, "record_type": "slide_oram_configuration_result", "component": "SLIDE-ORAM",
              "row": row, "parameters": parameters, "summary": summary, "command": command, "python": sys.version,
              "thread_basis": ("confirmed_single_thread_implementation" if table_threads == DEFAULT_TABLE_THREADS else
                               "explicit_verified_thread_count" if table_threads is not None else "effective_parallelism_unconfirmed"),
              "sources": [{"file": base.relative(Path(module)), "sha256": base.sha256(Path(module))}
                          for module in (__file__, Path(__file__).with_name("oram_log.py"), naming.__file__, base.__file__)],
              "runs": runs, "skipped_inputs": skipped,
              "table_contract": "SLIDE-ORAM baseline row; batch_hours is the display value, batch_seconds is retained for cross-method comparison; merge compatible dataset/batchsize/settings only; retain method/thread rows separately."}
    base.local_path(output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{dataset}, batch size {batchsize}: {summary['n_batches']} complete batches, {summary['n_runs']} run(s); "
          f"amortized batch {summary['batch_hours']:.6f} h, epoch {summary['epoch_hours']:.3f} h")
    print(f"Rehash: {summary['rehash_event_mean_seconds']:.3f} s/event / {summary['rehash_interval_batches']} batches "
          f"= {summary['rehash_amortized_seconds']:.3f} s/batch")
    print(f"Outputs: {naming.path_from_cwd(output)}/")
    if output_arg is None:
        print(f"For custom parameters or measurement windows, use --output-dir analysis/slide-oram/{dataset}/custom-name")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="case/separator-insensitive dataset name or alias")
    parser.add_argument("--runs-dir", default="SLIDE-ORAM/dataset/runs", help="relative to artifact root")
    parser.add_argument("--run", action="append", default=[], help="exact run folder name; repeat to select runs")
    parser.add_argument("--config", help="fallback KEY=VALUE config inside artifact, when effective metadata is absent")
    for option in ("threads", "batchsize", "bucketsize", "padsize"):
        description = ("filter recorded thread setting (e.g. OMP_NUM_THREADS); separate from table #Thd"
                       if option == "threads" else f"filter recorded {option}; does not modify the experiment")
        parser.add_argument("--" + option, action="append", default=[], help=description)
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--list", action="store_true", help="list configurations without writing results")
    parser.add_argument("--skip-batches", type=int, default=0, help="skip this many retained complete batches per run")
    parser.add_argument("--max-batches", type=int, help="at most this many subsequent complete batches per run")
    parser.add_argument("--rehash-seconds", type=float, help="measured per-event rehash calibration, if needed")
    parser.add_argument("--table-threads", type=int, default=DEFAULT_TABLE_THREADS,
                        help="effective thread count (default: 1 for the confirmed single-thread implementation); override only for a verified different implementation")
    parser.add_argument("--output-dir", help="relative to artifact root; use a named dataset subfolder for custom parameters")
    args = parser.parse_args(argv)
    dataset = naming.normalize_dataset(args.dataset)
    if args.table_threads is not None:
        positive_integer({"threads": args.table_threads}, "threads")
    filters = {}
    for key, values in (("threads", args.threads), ("Batchsize", args.batchsize), ("BUCKETSIZE", args.bucketsize), ("PADSIZE", args.padsize)):
        for value in values:
            normalized = naming.parameter_value(value)
            positive_integer({key: normalized}, key)
            naming.add_parameter(filters, key, normalized, "command line")
    for value in args.param:
        key, separator, raw = value.partition("=")
        if not separator:
            raise ValueError("--param requires KEY=VALUE")
        naming.add_parameter(filters, key, raw, "command line")
    runs, skipped = discover_runs(args.runs_dir, dataset, args.run, args.config, filters)
    for skipped_run in skipped:
        print(f"Skipped {skipped_run['input']}: {skipped_run['reason']}")
    if args.run and set(args.run) - {run["folder"] for run in runs}:
        raise ValueError("Some requested folders have no usable runs for this dataset.")
    if not runs:
        raise ValueError(f"No complete SLIDE-ORAM batches found for {dataset}.")
    if args.list:
        matches = [run for run in runs if all(run["parameters"].get(k) == v for k, v in filters.items())]
        if not matches:
            raise ValueError("No configuration matches.\n" + choices(runs))
        print(choices(matches, filters))
        return 0
    selected = select_runs(runs, filters)
    summary = summarize_runs(selected, selected[0]["parameters"], args.skip_batches, args.max_batches, args.rehash_seconds)
    write_results(selected, skipped, summary, dataset, args.output_dir, sys.argv if argv is None else argv, args.table_threads)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
