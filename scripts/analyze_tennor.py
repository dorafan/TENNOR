#!/usr/bin/env python3
"""Analyze one TENNOR configuration: amortized batch time, Table 2 row, breakdown."""

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
from tennor_log import parse_console


COMPONENTS = ("requester_s", "oht_s", "lsh_scan_s", "reqtrans_s", "fetcher_s", "updater_s")


def positive_integer(parameters, key):
    value = str(parameters.get(key, ""))
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"A positive {key} is required in run.json, config.csv, or the folder name.")
    return int(value)


def average(values):
    return statistics.mean(values)


def global_diagnostics(batches):
    """Global differences are diagnostic only; never substitute them for batch timers."""
    rows = []
    previous = None
    for batch in batches:
        current = batch.get("global_seconds")
        delta = None
        if current is not None:
            if (previous is None and batch["batch"] == 0 and batch.get("epoch") in (None, 0)
                    and abs(current - batch["reported_seconds"]) <= 1):
                delta = current
            elif (previous is not None and previous.get("global_seconds") is not None
                  and batch.get("epoch") == previous.get("epoch")
                  and batch.get("segment") == previous.get("segment")
                  and batch["batch"] == previous["batch"] + 1
                  and current >= previous["global_seconds"]):
                delta = current - previous["global_seconds"]
        rows.append(delta)
        previous = batch
    return rows


def summarize_runs(runs, parameters, skip_batches=0, max_batches=None, rehash_seconds=None):
    batchsize = positive_integer(parameters, "Batchsize")
    records = positive_integer(parameters, "totRecords")
    rehash = positive_integer(parameters, "Rehash")
    positive_integer(parameters, "threads")
    if rehash % batchsize or rehash < batchsize:
        raise ValueError("Rehash must be a positive multiple of Batchsize; otherwise the refresh "
                         "schedule needs explicit verification.")
    if records < batchsize:
        raise ValueError("totRecords must contain at least one complete batch.")
    if skip_batches < 0 or (max_batches is not None and max_batches < 1):
        raise ValueError("skip_batches must be nonnegative; max_batches must be positive.")
    if rehash_seconds is not None and (not math.isfinite(rehash_seconds) or rehash_seconds < 0):
        raise ValueError("--rehash-seconds must be finite and nonnegative.")
    windows = []
    excluded_runs = []
    for run in runs:
        all_batches = run["batches"]
        window = all_batches[skip_batches:None if max_batches is None else skip_batches + max_batches]
        if window:
            windows.append((run, window))
        else:
            excluded_runs.append(run["run_id"])
    if not windows:
        raise ValueError("No complete batches remain in the requested window.")

    # A short selected window may not contain a refresh. Calibrate only from
    # completed batches in the same selected configuration's contributing logs.
    events, rejected_events = [], []
    for run, _ in windows:
        for batch in run["batches"]:
            if not batch.get("rehash_present"):
                continue
            event = {"run_id": run["run_id"], "epoch": batch.get("epoch"), "batch": batch["batch"],
                     "seconds": batch.get("rehash_s"), "header_line": batch.get("header_line")}
            total, seconds = batch.get("reported_seconds"), batch.get("rehash_s")
            if (not batch.get("rehash_valid") or seconds is None or total is None
                    or not math.isfinite(seconds) or not math.isfinite(total)
                    or seconds < 0 or total < seconds):
                rejected_events.append(dict(event, reason="invalid rehash timing or rehash exceeds batch total"))
            else:
                events.append(event)
    if rehash_seconds is None and not events:
        raise ValueError("No complete, timed rehash event is available for amortization. "
                         "Include a run containing one, or supply a measured --rehash-seconds value.")
    event_mean = rehash_seconds if rehash_seconds is not None else average([event["seconds"] for event in events])
    interval = rehash // batchsize
    overhead = event_mean / interval
    per_run = []
    all_components_valid = True
    for run, window in windows:
        for batch in window:
            total = batch.get("reported_seconds")
            if total is None or not math.isfinite(total) or total < 0:
                raise ValueError(f"{run['run_id']}, batch {batch['batch']}: invalid printed batch duration.")
            if not batch.get("rehash_valid", True) or batch.get("rehash_s") is None:
                raise ValueError(f"{run['run_id']}, batch {batch['batch']}: rehash duration is incomplete.")
            if batch["reported_seconds"] < batch["rehash_s"]:
                raise ValueError(f"{run['run_id']}, batch {batch['batch']}: rehash exceeds total batch time.")
        raw = average([batch["reported_seconds"] for batch in window])
        core = average([batch["reported_seconds"] - batch["rehash_s"] for batch in window])
        valid = all(batch.get("breakdown_valid", False) and
                    all(batch.get(key) is not None for key in COMPONENTS) for batch in window)
        components = {key: average([batch[key] for batch in window]) for key in COMPONENTS} if valid else None
        if components is not None:
            components["other_s"] = core - sum(components[key] for key in ("requester_s", "fetcher_s", "updater_s"))
        all_components_valid &= valid
        deltas = global_diagnostics(run["batches"])
        window_deltas = deltas[skip_batches:None if max_batches is None else skip_batches + max_batches]
        global_mean = average(window_deltas) if all(value is not None for value in window_deltas) else None
        per_run.append({"run_id": run["run_id"], "n_batches": len(window),
                        "selected_batches": [{"epoch": batch.get("epoch"), "batch": batch["batch"],
                                              "header_line": batch.get("header_line")} for batch in window],
                        "raw_batch_seconds": raw, "core_batch_seconds": core,
                        "batch_seconds": core + overhead, "components": components,
                        "global_delta_mean_seconds": global_mean,
                        "global_minus_reported_mean_seconds": None if global_mean is None else global_mean - raw})
    mean_core = average([run["core_batch_seconds"] for run in per_run])
    mean_batch = mean_core + overhead
    components = ({key: average([run["components"][key] for run in per_run])
                   for key in (*COMPONENTS, "other_s")} if all_components_valid else None)
    batches_per_epoch = records // batchsize
    notes = ["Batch totals use the printed Training Batch timer; Global is diagnostic only.",
             "Calendar timestamp intervals have one-second resolution; readback retains its printed precision.",
             "Rehash calibration uses all complete batches in the contributing logs, including outside the selected window.",
             "Network initialization is excluded; batch 0 is not automatically treated as warm-up."]
    if excluded_runs:
        notes.append("Runs with no batches in the selected window do not contribute to the mean.")
    if rejected_events:
        notes.append("Invalid rehash calibration candidates were excluded; see rejected_rehash_events.")
    if not all_components_valid:
        notes.append("Breakdown unavailable: at least one selected complete batch lacks required timing boundaries.")
    if components is not None and components["other_s"] < 0:
        notes.append("Signed timing residual is negative; it is preserved, not clipped or redistributed.")
    return {"batch_seconds": mean_batch,
            "raw_batch_seconds": average([run["raw_batch_seconds"] for run in per_run]),
            "core_batch_seconds": mean_core, "rehash_amortized_seconds": overhead,
            "rehash_event_mean_seconds": event_mean, "rehash_event_count": len(events),
            "rehash_interval_batches": interval, "rehash_events": events,
            "rejected_rehash_events": rejected_events,
            "rehash_calibration_source": "explicit_override" if rehash_seconds is not None else "complete_batches_in_contributing_logs",
            "n_runs": len(per_run), "n_batches": sum(run["n_batches"] for run in per_run),
            "n_inputs": sum(run["n_batches"] for run in per_run) * batchsize,
            "batches_per_epoch": batches_per_epoch, "unused_epoch_records": records % batchsize,
            "epoch_hours": mean_batch * batches_per_epoch / 3600,
            "run_mean_std_seconds": statistics.stdev([run["batch_seconds"] for run in per_run]) if len(per_run) > 1 else None,
            "components": components, "run_summaries": per_run, "excluded_runs": excluded_runs,
            "window": {"skip_batches": skip_batches, "max_batches": max_batches},
            "averaging": "equal weight per independent run after averaging its selected complete batches",
            "notes": notes}


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


def case_file(directory, filename):
    matches = [path for path in directory.iterdir() if path.is_file() and path.name.casefold() == filename.casefold()]
    if len(matches) > 1:
        raise ValueError(f"{directory}: ambiguous filename variants for {filename}")
    return base.local_path(matches[0]) if matches else None


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
        # Historical TENNOR names use a bare thread count, e.g. wiki10_44.
        # Accept this only when the manifest independently confirms that count.
        end, _ = naming.dataset_prefix(run_dir.name)
        suffix = run_dir.name[end:].strip(" _-")
        legacy_threads = suffix.isdigit() and recorded.get("threads") == str(int(suffix))
        named = ({"parameters": {"threads": str(int(suffix))}, "repeat": None}
                 if legacy_threads else naming.parse_run_name(run_dir.name))
        for key, value in named["parameters"].items():
            if key in recorded and recorded[key] != value:
                raise ValueError(f"{run_dir.name}: folder {key}={value} conflicts with recorded {recorded[key]}")
            naming.add_parameter(parameters, key, value, run_dir.name)
        log = case_file(run_dir, "console.log")
        if log is None:
            skipped.append({"input": base.relative(run_dir), "reason": "no console.log"})
            continue
        raw = log.read_bytes()
        text = raw.decode("utf-8-sig")
        observed_threads = set(re.findall(r"\[(?:pass4|readback)\]\s+threads=(\d+)", text))
        if len(observed_threads) > 1:
            raise ValueError(f"{run_dir.name}: multiple runtime thread counts: {sorted(observed_threads)}")
        if observed_threads:
            naming.add_parameter(parameters, "threads", next(iter(observed_threads)), f"{run_dir.name}: console.log")
        parsed = parse_console(text)
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


def check_batch_reset_policy(runs, policy):
    ambiguous = [run for run in runs if any(item.get("kind") == "unlabelled_batch_reset" for item in run["audit"])]
    if ambiguous and policy != "resume":
        paths = ", ".join(run["run_id"] for run in ambiguous)
        raise ValueError("Batch numbers restart without Epoch headers in " + paths
                         + ". Strict reset checking is enabled. Use --batch-reset-policy resume "
                         "for TENNOR's global batch IDs, where repeated IDs are retries.")
    if ambiguous:
        print("Batch reset policy: treating unlabelled repeated batch IDs as resumed attempts; latest complete occurrence wins.")


def choices(runs, filters=None):
    keys, groups = naming.configuration_groups(runs)
    differing = [key for key in keys if len({run["parameters"].get(key) for run in runs}) > 1]
    lines = ["TENNOR configurations (thread count and batch size are separate dimensions):"]
    for index, (signature, members) in enumerate(groups.items(), 1):
        values = dict(signature)
        shown = differing or [key for key in ("threads", "Batchsize", "BUCKETSIZE", "Rehash") if key in values]
        detail = ", ".join(f"{key}={values.get(key) if values.get(key) is not None else 'UNKNOWN'}" for key in shown)
        lines.append(f"  [{index}] {detail}; runs={len(members)}; complete batches={sum(len(r['batches']) for r in members)}")
        for run in members:
            lines.append("      " + run["run_id"])
        interpreter = Path(sys.executable).absolute()
        executable = naming.path_from_cwd(interpreter) if base.ROOT in interpreter.parents else interpreter.as_posix()
        command = [executable, naming.path_from_cwd(Path(__file__).resolve()), "--dataset", members[0]["dataset"]]
        if members[0]["runs_directory"] != "TENNOR/dataset/runs":
            command += ["--runs-dir", members[0]["runs_directory"]]
        selected_filters = dict(filters or {})
        if differing and all(values[key] is not None for key in differing):
            selected_filters.update({key: values[key] for key in differing})
        else:
            for run in members:
                command += ["--run", run["folder"]]
        for key, value in selected_filters.items():
            flag = {"threads": "--threads", "Batchsize": "--batchsize", "BUCKETSIZE": "--bucketsize",
                    "PADSIZE": "--padsize"}.get(key)
            command += [flag, value] if flag else ["--param", f"{key}={value}"]
        lines.append("      Select: " + ("& " if os.name == "nt" else "") + shlex.join(command))
    return "\n".join(lines)


def select_runs(runs, filters):
    selected = [run for run in runs if all(run["parameters"].get(key) == value for key, value in filters.items())]
    if not selected:
        raise ValueError("No configuration matches.\n" + choices(runs))
    if len(naming.configuration_groups(selected)[1]) > 1:
        raise ValueError("Select one configuration before averaging. Different threads or batch sizes "
                         "are never averaged together.\n" + choices(selected, filters))
    repeats = [run["repeat"] for run in selected if run["repeat"] is not None]
    if len(repeats) != len(set(repeats)):
        raise ValueError("Duplicate repeat IDs in one configuration; select independent runs with --run.")
    return selected


def make_breakdown(summary, row, output):
    stem = f"{row['dataset']}_bs{row['batchsize']}_threads{row['threads']}_breakdown"
    if summary["components"] is None:
        # Refreshing a result must not leave an old figure beside a new table.
        for extension in ("pdf", "png"):
            path = base.local_path(output / f"{stem}.{extension}")
            if path.is_file():
                path.unlink()
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = summary["components"]
    total = [("Requester", values["requester_s"], "#1f77b4"),
             ("Fetcher", values["fetcher_s"], "#ff7f0e"),
             ("Updater", values["updater_s"], "#2ca02c")]
    fetcher = [("OHT", values["oht_s"], "#8c564b"),
               ("LSH scan", values["lsh_scan_s"], "#e377c2"),
               ("ReqTrans", values["reqtrans_s"], "#bcbd22")]
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42, "axes.titlesize": 14}):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.2), layout="constrained")
        ymax = max(sum(item[1] for item in total), sum(item[1] for item in fetcher), 1) * 1.12
        for ax, stages, title in zip(axes, (total, fetcher), ("Components", "Fetcher")):
            bottom = 0
            for label, seconds, color in stages:
                ax.bar([0], [seconds], bottom=[bottom], width=0.42, color=color, label=label)
                bottom += seconds
            ax.set(title=title, xlabel="Threads", ylabel="Time per batch (s)", ylim=(0, ymax), xlim=(-0.65, 0.65))
            ax.set_xticks([0], [str(row["threads"])])
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9, frameon=False)
        fig.suptitle(f"{base.DATASETS[row['dataset']][0]} · batch size {row['batchsize']}\n"
                     f"{row['n_runs']} run(s), {row['n_batches']} complete batches")
        generated = []
        for extension in ("pdf", "png"):
            path = base.local_path(output / f"{stem}.{extension}")
            fig.savefig(path, dpi=180)
            generated.append(base.relative(path))
        plt.close(fig)
    return generated


def write_results(runs, skipped, summary, dataset, output_arg, command, batch_reset_policy="resume"):
    parameters = runs[0]["parameters"]
    batchsize, threads = int(parameters["Batchsize"]), int(parameters["threads"])
    output = base.local_path(output_arg or f"analysis/tennor/{dataset}/bs{batchsize}/threads{threads}")
    for protected in ("TENNOR", "SLIDE-MP", "SLIDE-ORAM", "scripts"):
        path = base.ROOT / protected
        if output == path or path in output.parents:
            raise ValueError("Choose an analysis output directory outside components and scripts.")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(base.local_path(output / ".matplotlib"))
    row = {"dataset": dataset, "method": "TENNOR", "threads": threads, "batchsize": batchsize,
           "batch_seconds": summary["batch_seconds"], "speedup": None, "epoch_hours": summary["epoch_hours"],
           "n_runs": summary["n_runs"], "n_batches": summary["n_batches"], "n_inputs": summary["n_inputs"],
           "batches_per_epoch": summary["batches_per_epoch"], "raw_batch_seconds": summary["raw_batch_seconds"],
           "rehash_amortized_seconds": summary["rehash_amortized_seconds"],
           "timing_basis": "printed_batch_timer_with_rehash_amortization"}
    base.write_csv(output / "table_row.csv", list(row), [row])
    markdown = (f"| Dataset | Method | #Thd | Batch(s) | SU | Ep.(h) |\n"
                f"| --- | --- | ---: | ---: | ---: | ---: |\n"
                f"| {base.DATASETS[dataset][0]} | TENNOR | {threads} | {row['batch_seconds']:.3f} | N/A | {row['epoch_hours']:.3f} |\n\n"
                f"Batch size: {batchsize}. {row['n_runs']} run(s), {row['n_batches']} complete batches "
                f"({row['n_inputs']} observed inputs). Epoch time is an extrapolation using {row['batches_per_epoch']} batches.\n\n"
                "SU is unavailable until a matching SLIDE-ORAM measurement is supplied. "
                "Compare thread counts only within the same dataset, batch size, and other training settings.\n")
    base.local_path(output / "table_row.md").write_text(markdown, encoding="utf-8")
    selected = {(run["run_id"], batch["epoch"], batch["batch"], batch["header_line"])
                for run in summary["run_summaries"] for batch in run["selected_batches"]}
    details = []
    for run in runs:
        deltas = global_diagnostics(run["batches"])
        for batch, delta in zip(run["batches"], deltas):
            identity = (run["run_id"], batch.get("epoch"), batch["batch"], batch.get("header_line"))
            details.append(dict(batch, run_id=run["run_id"], global_delta_seconds=delta, selected=identity in selected))
    columns = ["run_id", "epoch", "batch", "segment", "occurrence", "header_line", "end_line", "selected",
               "reported_seconds", "global_seconds", "global_delta_seconds", "rehash_present", "rehash_valid", "rehash_s",
               *COMPONENTS, "readback_total_s", "readback_clear_s", "readback_wall_s", "breakdown_valid", "errors"]
    base.write_csv(output / "batches.csv", columns, details)
    base.write_csv(output / "run_summary.csv", ["run_id", "n_batches", "raw_batch_seconds", "core_batch_seconds",
                                                "batch_seconds", "global_delta_mean_seconds", "global_minus_reported_mean_seconds"], summary["run_summaries"])
    plots = make_breakdown(summary, row, output)
    result = {"schema_version": 1, "record_type": "tennor_configuration_result", "component": "TENNOR",
              "row": row, "parameters": parameters, "summary": summary, "command": command,
              "python": sys.version,
              "batch_reset_policy": batch_reset_policy,
              "sources": [{"file": base.relative(Path(module)), "sha256": base.sha256(Path(module))}
                          for module in (__file__, Path(__file__).with_name("tennor_log.py"), naming.__file__, base.__file__)],
              "runs": runs, "skipped_inputs": skipped, "plots": plots,
              "table_contract": "Merge by dataset+batchsize+compatible settings; retain threads as a separate row; SU needs same-dataset/same-batchsize ORAM."}
    base.local_path(output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{dataset}, batch size {batchsize}, threads {threads}: {summary['n_batches']} complete batches, "
          f"{summary['n_runs']} run(s); amortized batch {summary['batch_seconds']:.3f} s, epoch {summary['epoch_hours']:.3f} h")
    print(f"Rehash: {summary['rehash_event_mean_seconds']:.3f} s/event / {summary['rehash_interval_batches']} batches "
          f"= {summary['rehash_amortized_seconds']:.3f} s/batch")
    if summary["components"] is None:
        print("Breakdown not generated: required timing boundaries are missing; see result.json.")
    print(f"Outputs: {naming.path_from_cwd(output)}/")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runs-dir", default="TENNOR/dataset/runs", help="relative to the artifact root")
    parser.add_argument("--run", action="append", default=[], help="exact run folder name; repeat to select runs")
    parser.add_argument("--config", help="fallback KEY=VALUE config inside the artifact for logs without effective metadata")
    for option in ("threads", "batchsize", "bucketsize", "padsize"):
        parser.add_argument("--" + option, action="append", default=[])
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--skip-batches", type=int, default=0, help="skip this many complete batches per run")
    parser.add_argument("--max-batches", type=int, help="at most this many subsequent complete batches per run")
    parser.add_argument("--rehash-seconds", type=float, help="explicit measured per-event calibration, if needed")
    parser.add_argument("--batch-reset-policy", choices=("error", "resume"), default="resume",
                        help="global batch IDs: repeated IDs are retries by default; error enables strict reset checks")
    parser.add_argument("--output-dir", help="relative to artifact root; default separate dataset/batchsize/threads")
    args = parser.parse_args(argv)
    dataset = naming.normalize_dataset(args.dataset)
    filters = {}
    for key, values in (("threads", args.threads), ("Batchsize", args.batchsize),
                        ("BUCKETSIZE", args.bucketsize), ("PADSIZE", args.padsize)):
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
        raise ValueError(f"No complete TENNOR batches found for {dataset}.")
    if args.list:
        matches = [run for run in runs if all(run["parameters"].get(k) == v for k, v in filters.items())]
        if not matches:
            raise ValueError("No configuration matches.\n" + choices(runs))
        print(choices(matches, filters))
        return 0
    selected = select_runs(runs, filters)
    check_batch_reset_policy(selected, args.batch_reset_policy)
    summary = summarize_runs(selected, selected[0]["parameters"], args.skip_batches, args.max_batches, args.rehash_seconds)
    write_results(selected, skipped, summary, dataset, args.output_dir, sys.argv if argv is None else argv, args.batch_reset_policy)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
