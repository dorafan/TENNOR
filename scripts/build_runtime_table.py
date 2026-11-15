#!/usr/bin/env python3
"""Assemble TENNOR and SLIDE-ORAM result rows into dataset-grouped runtime tables."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import analysis_common as base
import analyze_slide_mp as naming
from runtime_table_data import load_result


MATCH_KEYS = ("Batchsize", "InputDim", "numLayer", "sizesOfLayers", "totRecords",
              "totRecordsTest", "Lr", "Sparsity", "Rehash", "Rebuild")
OUTPUT_ROOT = "analysis/runtime"
DATASET_ORDER = ("wiki10", "amzprime", "wiki325", "amazon670")


def dataset_order(dataset):
    return (DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER), dataset)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def duplicate_key(record):
    """Only collapse copies of the same measured result, never independent runs."""
    metrics = {key: record.get(key) for key in
               ("method", "dataset", "threads", "batchsize", "batch_seconds", "batch_hours", "epoch_hours",
                "n_runs", "n_batches", "n_inputs", "batches_per_epoch", "timing_basis", "window", "parameters")}
    metrics["measurement_key"] = record.get("measurement_key") or record["source"]
    return canonical(metrics)


def algorithm_profile(record):
    # Thread count is the independent scaling variable; every other setting
    # within one method must be the same before placing its rows together.
    return canonical({key: value for key, value in record["parameters"].items() if key != "threads"})


def candidates_error(reason, records):
    choices = "\n".join("  --result " + json.dumps(record["source"]) for record in records)
    return ValueError(reason + ". Select the intended result files explicitly with repeated --result options:\n" + choices)


def build_groups(records):
    unique, audit = {}, []
    for source in sorted(records, key=lambda record: record["source"]):
        record = dict(source, duplicate_sources=[])
        key = duplicate_key(record)
        if key in unique:
            prior = unique[key]
            prior["duplicate_sources"].append(record["source"])
            audit.append({"kind": "duplicate_result", "source": record["source"], "kept_source": prior["source"]})
        else:
            unique[key] = record
    buckets = defaultdict(list)
    for record in unique.values():
        signature = (record["dataset"], record["batchsize"],
                     tuple((key, record["parameters"].get(key)) for key in MATCH_KEYS))
        buckets[signature].append(record)
    ordered = sorted(buckets.items(), key=lambda item:
                     (dataset_order(item[0][0]), item[0][1], canonical(item[0][2])))
    groups = []
    for (dataset, batchsize, signature), members in ordered:
        parameters = dict(signature)
        missing = [key for key, value in signature if value is None]
        oram = [record for record in members if record["method"] == "SLIDE-ORAM"]
        tennor = [record for record in members if record["method"] == "TENNOR"]
        for method, records_for_method in (("SLIDE-ORAM", oram), ("TENNOR", tennor)):
            if len({algorithm_profile(record) for record in records_for_method}) > 1:
                raise candidates_error(f"{dataset}, batchsize {batchsize}: multiple {method} algorithm configurations for one training task", records_for_method)
        if len(oram) > 1:
            raise candidates_error(f"{dataset}, batchsize {batchsize}: multiple SLIDE-ORAM baseline measurements", oram)
        by_threads = defaultdict(list)
        for record in tennor:
            by_threads[record["threads"]].append(record)
        for threads, same_threads in by_threads.items():
            if len(same_threads) > 1:
                raise candidates_error(f"{dataset}, batchsize {batchsize}: multiple TENNOR measurements at threads={threads}", same_threads)
        baseline = oram[0] if oram and not missing else None
        rows = []
        for record in oram + sorted(tennor, key=lambda value: (value["threads"] is None, value["threads"] or 0)):
            row = dict(record)
            if record["method"] == "SLIDE-ORAM":
                row.update(speedup=1.0, baseline_source=record["source"], status="baseline")
            elif baseline is not None:
                row.update(speedup=baseline["batch_seconds"] / record["batch_seconds"],
                           baseline_source=baseline["source"], status="matched")
            else:
                row.update(speedup=None, baseline_source=None,
                           status="incomplete_configuration" if missing else "no_baseline")
            rows.append(row)
        status = ("incomplete_configuration" if missing else
                  "matched" if oram and tennor else "no_baseline" if tennor else "baseline_only")
        group = {"dataset": dataset, "batchsize": batchsize,
                 "match_parameters": parameters, "missing_keys": missing, "rows": rows, "status": status}
        if status != "matched":
            audit.append({"kind": status, "dataset": dataset, "batchsize": group["batchsize"],
                          "missing_keys": missing, "sources": [row["source"] for row in rows]})
        groups.append(group)
    return groups, audit


def collect_paths(tennor_dir, oram_dir, explicit):
    paths = {}
    if explicit:
        for raw in explicit:
            path = base.local_path(raw)
            if path.is_dir():
                path /= "result.json"
            elif path.name == "table_row.csv":
                path = path.with_name("result.json")
            if not path.is_file():
                raise ValueError(f"Result file not found: {path}")
            paths[base.local_path(path)] = None
    else:
        for raw, method in ((tennor_dir, "TENNOR"), (oram_dir, "SLIDE-ORAM")):
            directory = base.local_path(raw)
            if not directory.exists():
                continue
            if not directory.is_dir():
                raise ValueError(f"Expected an analysis directory: {directory}")
            for path in sorted(directory.rglob("result.json")):
                path = base.local_path(path)
                if path in paths and paths[path] != method:
                    raise ValueError("TENNOR and SLIDE-ORAM input directories overlap.")
                paths[path] = method
    if not paths:
        raise ValueError("No result.json files found. Run the component analyzers first, or supply --result PATH.")
    return paths


def select_records(records, datasets=(), batchsizes=(), filters=None):
    filters = filters or {}
    return [record for record in records
            if (not datasets or record["dataset"] in datasets)
            and (not batchsizes or record["batchsize"] in batchsizes)
            and all(record["parameters"].get(key) == value for key, value in filters.items())]


def display_batch(row):
    if row["method"] == "SLIDE-ORAM":
        return f"{row['batch_hours']:,.6f} h"
    return f"{row['batch_seconds']:,.3f} s"


def display_threads(row):
    return str(row["threads"]) if row["threads"] is not None else "unknown"


def display_method(row):
    return "SLIDE+ORAM" if row["method"] == "SLIDE-ORAM" else row["method"]


def group_details(group, peers):
    """Readable distinctions when a dataset/batch size has several task settings."""
    same_size = [peer for peer in peers if peer["batchsize"] == group["batchsize"]]
    differing = [key for key in MATCH_KEYS if key != "Batchsize"
                 and len({peer["match_parameters"][key] for peer in same_size}) > 1]
    return ", ".join(f"{key}={group['match_parameters'][key] or 'UNKNOWN'}" for key in differing)


def render_markdown(groups):
    lines = ["# Runtime comparison", "",
             "SU is relative to the matching SLIDE+ORAM result and is computed in seconds. "
             "Batch values display hours for SLIDE+ORAM and seconds for TENNOR. Epoch time is extrapolated.", "",
             "Rows describe the available measurements; sample counts are reported below each table. "
             "Method-specific algorithm parameters are preserved in result.json."]
    datasets = sorted({group["dataset"] for group in groups}, key=dataset_order)
    for dataset in datasets:
        peers = [group for group in groups if group["dataset"] == dataset]
        title = base.DATASETS[dataset][0]
        lines += ["", f"## {title}"]
        for group in peers:
            details = group_details(group, peers)
            lines += ["", f"### Batch size {group['batchsize']}" + (f" — {details}" if details else ""), "",
                      "| Dataset | Method | #Thd | Batch | SU | Ep.(h) |",
                      "| --- | --- | ---: | ---: | ---: | ---: |"]
            for row in group["rows"]:
                speedup = "N/A" if row["speedup"] is None else f"{row['speedup']:.3f}×"
                lines.append(f"| {title} | {display_method(row)} | {display_threads(row)} | "
                             f"{display_batch(row)} | {speedup} | {row['epoch_hours']:,.3f} |")
            lines += ["", "Observed samples:", ""]
            for row in group["rows"]:
                lines.append(f"- {display_method(row)}, #Thd={display_threads(row)}: "
                             f"{row['n_runs']} run(s), {row['n_batches']} complete batches, {row['n_inputs']} inputs.")
            if group["missing_keys"]:
                lines += ["", "TENNOR SU is unavailable: missing matching fields " + ", ".join(group["missing_keys"]) + "."]
            elif group["status"] == "no_baseline":
                lines += ["", "TENNOR SU is unavailable: no SLIDE+ORAM result matches this training task."]
            elif group["status"] == "baseline_only":
                lines += ["", "No TENNOR result is available for this training task."]
    if any(row["threads"] is None for group in groups for row in group["rows"]):
        lines += ["", "Unknown effective thread counts remain unknown; requested OpenMP settings are not substituted."]
    return "\n".join(lines) + "\n"


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(character, character) for character in str(value))


def render_latex(groups):
    lines = ["% Generated table fragments; no additional LaTeX packages are required.",
             "% SU uses seconds; the Batch column explicitly labels seconds or hours."]
    for group in groups:
        title = base.DATASETS[group["dataset"]][0]
        peers = [peer for peer in groups if peer["dataset"] == group["dataset"]]
        detail = group_details(group, peers)
        caption = f"Runtime comparison on {title}, batch size {group['batchsize']}"
        if detail:
            caption += "; " + detail
        lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{tabular}{lllrrr}", r"\hline",
                  r"Dataset & Method & \#Thd & Batch & SU & Ep.(h) \\", r"\hline"]
        for row in group["rows"]:
            speedup = "N/A" if row["speedup"] is None else f"${row['speedup']:.3f}\\times$"
            cells = [tex_escape(title), tex_escape(display_method(row)), tex_escape(display_threads(row)),
                     tex_escape(display_batch(row)), speedup, f"{row['epoch_hours']:,.3f}"]
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}", "\\caption{" + tex_escape(caption) + "}", r"\end{table}"]
        for row in group["rows"]:
            lines.append(f"% {row['method']} threads={display_threads(row)}: "
                         f"{row['n_runs']} runs, {row['n_batches']} complete batches, {row['n_inputs']} inputs.")
        if group["status"] != "matched":
            lines.append("% Matching status: " + group["status"])
        lines.append("")
    return "\n".join(lines)


def flat_rows(groups):
    result = []
    group_numbers = defaultdict(int)
    for group in groups:
        group_numbers[group["dataset"]] += 1
        index = group_numbers[group["dataset"]]
        for row in group["rows"]:
            result.append({"dataset": group["dataset"], "comparison_group": index,
                           "method": row["method"], "threads": row["threads"], "batchsize": row["batchsize"],
                           "batch_seconds": row["batch_seconds"], "batch_hours": row["batch_hours"],
                           "batch_display": display_batch(row),
                           "batch_display_unit": "h" if row["method"] == "SLIDE-ORAM" else "s",
                           "speedup": row["speedup"], "epoch_hours": row["epoch_hours"],
                           "n_runs": row["n_runs"], "n_batches": row["n_batches"], "n_inputs": row["n_inputs"],
                           "status": row["status"], "baseline_source": row["baseline_source"],
                           "source": row["source"], "csv_source": row["csv_source"],
                           "matching_parameters": canonical(group["match_parameters"])})
    return result


def write_tables(groups, audit, output_arg, command):
    root = base.local_path(OUTPUT_ROOT)
    output = base.local_path(output_arg or OUTPUT_ROOT)
    if output != root and root not in output.parents:
        raise ValueError(f"Table output must be {OUTPUT_ROOT} or one of its subdirectories; component results are preserved.")
    sources = [{"file": base.relative(Path(module)), "sha256": base.sha256(Path(module))}
               for module in (__file__, Path(__file__).with_name("runtime_table_data.py"), naming.__file__, base.__file__)]
    batchsizes = sorted({group["batchsize"] for group in groups})
    outputs = []
    for batchsize in batchsizes:
        batch_groups = [group for group in groups if group["batchsize"] == batchsize]
        batch_output = base.local_path(output / f"bs{batchsize}")
        datasets = sorted({group["dataset"] for group in batch_groups}, key=dataset_order)
        destinations = [(batch_output, batch_groups)] + [
            (batch_output / dataset, [group for group in batch_groups if group["dataset"] == dataset])
            for dataset in datasets]
        artifacts = []
        for directory, selected in destinations:
            directory = base.local_path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            rows = flat_rows(selected)
            base.write_csv(directory / "runtime_table.csv", list(rows[0]), rows)
            base.local_path(directory / "runtime_table.md").write_text(render_markdown(selected), encoding="utf-8")
            base.local_path(directory / "runtime_table.tex").write_text(render_latex(selected), encoding="utf-8")
            artifacts += [base.relative(directory / f"runtime_table.{extension}") for extension in ("csv", "md", "tex")]
        selected_sources = {row["source"] for group in batch_groups for row in group["rows"]}
        batch_audit = [item for item in audit if item.get("batchsize") == batchsize
                       or item.get("kept_source") in selected_sources]
        report = {"schema_version": 2, "record_type": "runtime_comparison_table", "command": command,
                  "batchsize": batchsize, "match_keys": list(MATCH_KEYS),
                  "matching_rule": "same dataset and recorded training task; method-specific algorithm parameters may differ",
                  "speedup_rule": "matching SLIDE-ORAM batch_seconds / TENNOR batch_seconds; no averaging of summary rows",
                  "groups": batch_groups, "audit": batch_audit, "artifacts": artifacts, "sources": sources}
        base.local_path(batch_output / "result.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        outputs.append(batch_output)
    print(f"Generated {sum(len(group['rows']) for group in groups)} rows in {len(groups)} training-task groups "
          f"across {len({group['dataset'] for group in groups})} dataset(s) and {len(batchsizes)} batch size(s).")
    for item in audit:
        if item["kind"] in ("no_baseline", "incomplete_configuration", "baseline_only"):
            print(f"{item['dataset']}, batchsize {item['batchsize']}: {item['kind']}")
        elif item["kind"] == "duplicate_result":
            print(f"Duplicate result copy omitted: {item['source']}")
    for directory in outputs:
        print(f"Outputs: {naming.path_from_cwd(directory)}/")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], help="dataset name/alias; repeat for multiple datasets; default all")
    parser.add_argument("--batchsize", action="append", type=int, default=[], help="repeat to include several batch sizes")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="filter a shared recorded task setting")
    parser.add_argument("--tennor-dir", default="analysis/tennor", help="artifact-root-relative analysis directory")
    parser.add_argument("--oram-dir", default="analysis/slide-oram", help="artifact-root-relative analysis directory")
    parser.add_argument("--result", action="append", default=[], help="select exact result.json/table_row.csv paths or result directories; replaces default discovery")
    parser.add_argument("--list", action="store_true", help="show available rows without merging or writing")
    parser.add_argument("--output-dir", help=f"output parent within {OUTPUT_ROOT}, relative to artifact root; bs<N> is appended automatically")
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.batchsize):
        raise ValueError("--batchsize must be positive")
    datasets = {naming.normalize_dataset(value) for value in args.dataset}
    filters = {}
    for raw in args.param:
        key, separator, value = raw.partition("=")
        if not separator:
            raise ValueError("--param requires KEY=VALUE")
        naming.add_parameter(filters, key, value, "command line")
    paths = collect_paths(args.tennor_dir, args.oram_dir, args.result)
    records = [load_result(path, expected_method=method) for path, method in paths.items()]
    records = select_records(records, datasets, set(args.batchsize), filters)
    if not records:
        raise ValueError("No result rows match the requested dataset, batch size, and parameter filters.")
    if args.list:
        for row in sorted(records, key=lambda value: (dataset_order(value["dataset"]), value["batchsize"], value["method"], value["threads"] or 0)):
            print(f"{row['dataset']}: {row['method']}, batchsize={row['batchsize']}, threads={display_threads(row)}, "
                  f"batch={display_batch(row)}, runs={row['n_runs']}, batches={row['n_batches']}\n  {row['source']}")
        return 0
    groups, audit = build_groups(records)
    write_tables(groups, audit, args.output_dir, sys.argv if argv is None else argv)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
