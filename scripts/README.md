# Result analysis scripts

These scripts analyze the three evaluated claims: TENNOR runtime and breakdown, SLIDE-ORAM runtime, and MP-WTA accuracy. They read recorded results without rerunning experiments or loading training datasets. The combined runtime table compares compatible TENNOR and SLIDE-ORAM measurements.

## Setup and paths

**Run every command below from the artifact root**, the directory containing `scripts/`, `TENNOR/`, `SLIDE-ORAM/`, `SLIDE-MP/`, and `SLIDE/`. Complete the shared installation once, then activate Python before analysis:

```bash
bash install_dependencies.sh
source .venv-ae/bin/activate
```

The installer targets Ubuntu 24.04 and creates `.venv-ae/` with the analysis dependencies. Reuse an existing installation; activate the environment once per terminal session. Plotting works without a display server. To collect new results, follow the component instructions: [TENNOR](../TENNOR/README.md), [SLIDE-ORAM](../SLIDE-ORAM/README.md), [SLIDE-MP](../SLIDE-MP/README.md), and [SLIDE WTA](../SLIDE/README.md).

All explicit relative path options, including `--runs-dir`, `--config`, `--wta-log`, `--wta-config`, `--result`, and `--output-dir`, resolve from the **artifact root**, regardless of the current working directory. Paths must remain inside the artifact; output directories must also be outside the component and `scripts/` directories, including `SLIDE/`. Analysis overwrites existing files at its output destination. Preserve different selections or measurement windows with a separate `--output-dir`, such as `analysis/tennor/wiki10/review-a`, `analysis/slide-oram/wiki10/review-a`, or `analysis/slide-mp/wiki10/review-a`.

## Included scripts

| Script | Role |
| --- | --- |
| [`analyze_tennor.py`](analyze_tennor.py) | TENNOR batch/epoch timing and two-panel breakdown figure |
| [`analyze_slide_oram.py`](analyze_slide_oram.py) | SLIDE-ORAM batch/epoch timing |
| [`analyze_slide_mp.py`](analyze_slide_mp.py) | MP-WTA/WTA accuracy comparison, explicit historical-reference mode, and dataset/configuration helpers |
| [`wta_logs.py`](wta_logs.py) | WTA log/configuration loading and evaluation-scope metadata |
| [`build_runtime_table.py`](build_runtime_table.py) | Combined runtime tables and speedups |
| [`tennor_log.py`](tennor_log.py) | TENNOR console-log parser |
| [`oram_log.py`](oram_log.py) | SLIDE-ORAM timestamped console-log parser |
| [`analysis_common.py`](analysis_common.py) | Shared paths, metadata, accuracy parsing, statistics, and plotting |
| [`runtime_table_data.py`](runtime_table_data.py) | Runtime-result loading, validation, and compatibility checks |

Keep these nine files together in `scripts/`. `analyze_tennor.py`, `analyze_slide_oram.py`, `analyze_slide_mp.py`, and `build_runtime_table.py` are command-line entry points; the remaining files are imported automatically. Use an entry point's `--help` for its full option list.

## Select recorded runs

Each analyzer requires `--dataset` and summarizes **one configuration at a time**. Start with `--list` to inspect available runs without writing outputs. Use the actual run names shown there; the examples below match the names in the component instructions.

| Dataset | `--dataset` | Other accepted names |
| --- | --- | --- |
| Wiki10-31K | `wiki10` | `wiki10-31k` |
| Wiki-325K | `wiki325` | `WikiLSHTC` |
| Amazon-670K | `amazon670` | `amazon670k`, `amz` |
| Amazon-Sub | `amzprime` | `amazonsub`, `amazonprime` |

Output paths use the canonical names in the middle column. Matching ignores case and separators. Only datasets with available logs can be analyzed.

`--run NAME` selects an exact folder under the component's `dataset/runs/`; repeat it to average independent runs with matching settings. `--param KEY=VALUE` filters recorded parameters. Filters select existing data and never modify experiments. Different configurations are not averaged together, and missing settings are not guessed. Keep logs with their original `run.json`; updated default configurations do not describe older runs.

For new experiments, use explicit names such as `amzprime-threads44-batchsize32-repeat1` for TENNOR or `amzprime-repeat1` for MP-WTA, with a distinct `repeatN` for each independent run of a configuration. The current analyzers do not accept arbitrary suffixes such as `startup1`; `--run` selects a folder but does not bypass name validation. See [Analyzing multiple experiments](#analyzing-multiple-experiments) for separate and averaged analyses.

## TENNOR runtime and breakdown

**Input:** `TENNOR/dataset/runs/<name>/console.log` and its matching `run.json`.

```bash
python scripts/analyze_tennor.py --dataset wiki10 --list
python scripts/analyze_tennor.py --dataset wiki10 --threads 8 --batchsize 32 --run wiki10-threads8-batchsize32-repeat1
```

The analyzer uses the printed `Training Batch ... takes ... seconds` timer for each complete batch. It produces amortized batch time, extrapolated epoch time, and Requester/Fetcher/Updater and internal Fetcher breakdowns. By default, repeated global batch IDs are treated as retries, retaining the latest complete occurrence; `--batch-reset-policy error` rejects resets instead.

**Output directory:** `analysis/tennor/<dataset>/bs<size>/threads<count>/`. In addition to the runtime files below, it writes `<dataset>_bs<size>_threads<count>_breakdown.pdf` and `.png`. The two panels show Requester/Fetcher/Updater and OHT/LSH scan/ReqTrans. Rehash and residual time are excluded from the panels, so their sums can differ from the table total. Missing component timings prevent figure generation even when batch timing is available.

## SLIDE-ORAM runtime

**Input:** `SLIDE-ORAM/dataset/runs/<name>/console.log`, timestamped by the launcher, and its matching `run.json`.

```bash
python scripts/analyze_slide_oram.py --dataset wiki10 --list
python scripts/analyze_slide_oram.py --dataset wiki10 --batchsize 32 --run wiki10-threads1-batchsize32-repeat1
```

The analyzer measures intervals between consecutive batch-start timestamps and checks that the batch's work is complete. A batch needs the next consecutive batch-start boundary; the final batch is excluded when that boundary is absent, even after successful process exit. `training.log` is not the timing input.

**Output directory:** `analysis/slide-oram/<dataset>/bs<size>/threads1/`. It writes the runtime files below without a figure. The supplied implementation has **one effective thread**. Its optional `--threads` filter selects the recorded OpenMP setting and does not change the effective thread label.

## Runtime outputs and measurement rules

Both runtime analyzers produce:

| Output | Contents |
| --- | --- |
| `table_row.csv`, `table_row.md` | Amortized batch time, extrapolated epoch time, and sample counts |
| `batches.csv` | Batch measurements and selection details |
| `run_summary.csv` | Per-run statistics |
| `result.json` | Recorded parameters, timing units, source hashes, selected batches, and diagnostics |

The default window includes all complete batches, including batch 0; initialization and unfinished tails are excluded. `--skip-batches N` and `--max-batches N` select complete batches in file order **per run**; they do not stop training. Batch timing is averaged within each run, then contributing runs receive equal weight.

Observed rehash cost is removed from the batch timings and replaced with an amortized cost: mean measured rehash-event seconds divided by `Rehash / Batchsize`. Calibration uses valid events in complete batches of contributing logs, including events outside the selected window. If none is available, supply a separately measured per-event cost with `--rehash-seconds VALUE`; missing cost is not assumed to be zero. `Rehash` must be a positive multiple of `Batchsize`.

**Epoch time is an extrapolation:** `epoch_hours = batch_seconds * floor(totRecords / Batchsize) / 3600`. TENNOR displays batch time in seconds; SLIDE-ORAM displays it in hours. Both retain seconds in their structured results and report epoch time in hours.

For legacy runtime logs without effective configuration metadata, a matching run-local `config.csv` or explicit `--config PATH` can supply the original settings. Use the configuration that actually produced the log.

## MP-WTA accuracy

**MP-WTA input:** `SLIDE-MP/dataset/runs/<name>/training.log` and its matching `run.json`. A custom `logFile` recorded by the launcher is also supported. **WTA input:** the new SLIDE training log and the CSV used to produce it. Follow the [SLIDE guide](../SLIDE/README.md) to collect WTA results before comparing them.

```bash
python scripts/analyze_slide_mp.py --dataset wiki10 --list
python scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1
```

`--list` lists MP-WTA configurations without requiring WTA logs or writing outputs. Analysis defaults to these WTA input/configuration pairs:

| Dataset | Default WTA log | Default WTA configuration |
| --- | --- | --- |
| `wiki10` | `SLIDE/dataset/Wiki10/training_wta.log` | `SLIDE/SLIDE/Config_wiki10_wta.csv` |
| `amzprime` | `SLIDE/dataset/AmzPrime/training_wta.log` | `SLIDE/SLIDE/Config_amzprime_wta.csv` |
| `wiki325` | `SLIDE/dataset/Wiki325/training_wta.log` | `SLIDE/SLIDE/Config_wiki325_wta.csv` |
| `amazon670` | `SLIDE/dataset/Amazon/training_wta.log` | `SLIDE/SLIDE/Config_amz_wta.csv` |

Missing or unusable WTA inputs cause an error; they are never silently replaced by the historical curve. To select a preserved WTA log explicitly:

```bash
python scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1 --wta-log SLIDE/dataset/Wiki10/training_wta.log --wta-config SLIDE/SLIDE/Config_wiki10_wta.csv --output-dir analysis/slide-mp/wiki10/measured-comparison
```

For independent repetitions, repeat `--run` for MP-WTA and `--wta-log` for WTA. Each method is aggregated separately; the numbers of runs need not match. For example, after preserving two independent WTA runs under the paths shown:

```bash
python scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1 --run wiki10-repeat2 --wta-log SLIDE/dataset/Wiki10/repeat1/training_wta.log --wta-log SLIDE/dataset/Wiki10/repeat2/training_wta.log --wta-config SLIDE/SLIDE/Config_wiki10_wta.csv --output-dir analysis/slide-mp/wiki10/measured-mean-r1-r2
```

One `--wta-config` applies to all selected WTA logs. Alternatively, repeat it once per WTA log in the same order. If omitted, the dataset's default CSV is used. **Supply the actual configuration used for each log**, especially if you changed `Epoch`, batch size, or model parameters; the CSV is evidence supplied by the user, not metadata embedded in the log. Preserve a copy with each run. Different WTA configurations cannot be averaged together, and duplicate paths or copied WTA log contents are rejected.

Collect repetitions on the same machine with the same execution environment. The SLIDE-MP launcher leaves thread selection to OpenMP and records the runtime-default policy without inventing a numeric thread count. Select these runs by name; existing thread metadata in older runs is still honored. The upstream WTA runner inherits OpenMP settings, so retain those settings with its logs.

For MP-WTA, the analyzer keeps the last evaluation in the latest complete epoch occurrence. A subsequent header with a strictly larger epoch number confirms a nonterminal epoch; the final epoch requires successful-completion metadata (`status=completed`, `returncode=0`, and configured `Epoch` equal to its zero-based epoch number plus one). For WTA, which has no launcher `run.json`, the analyzer requires an intact final evaluation row at iteration `(epoch + 1) * floor(totRecords / Batchsize)`, with the epoch inside the CSV's configured range. This retains the final WTA evaluation without assuming that checkpoint saving or the whole process completed successfully. Mid-epoch or malformed tails are excluded.

`Epoch 0` is the first trained epoch, and P@1 remains on the 0–1 scale. Each contributing run has equal weight per epoch within its method. Missing epochs are not filled, and plot lines break at gaps. Actual run counts are reported, with sample standard deviation only when at least two runs contribute.

**New SLIDE and SLIDE-MP logs use all complete test batches at every epoch end.** The analyzer defaults to `--wta-evaluation full`, matching the local SLIDE `main.cpp` change. Recompile SLIDE before collecting new logs; the CSV and log formats are unchanged. Each evaluation includes `floor(totRecordsTest / Batchsize) * Batchsize` samples, excluding the final incomplete batch. `comparison.csv` gives the difference **MP-WTA minus WTA** only where both methods have known, equal evaluated sample counts; otherwise the difference is blank. Matching counts describe the configured evaluation prefix, not independent verification of dataset contents. Keep the datasets and evaluation settings consistent when interpreting accuracy.

For preserved logs from the **unmodified upstream schedule**, explicitly select `--wta-evaluation legacy50`: non-final epochs used the first 50 complete test batches, and only the final epoch used all complete test batches. This mode retains the coverage distinction in the outputs and figure. The log format cannot distinguish the two schedules; use the mode that matches the executable which produced the logs, and do not combine logs from different schedules in one invocation. For example, with the old log and its actual CSV preserved at these paths:

```bash
python scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1 --wta-log SLIDE/dataset/Wiki10/legacy/training_wta.log --wta-config SLIDE/dataset/Wiki10/legacy/config.csv --wta-evaluation legacy50 --output-dir analysis/slide-mp/wiki10/legacy-comparison
```

To reproduce the older analysis using the embedded **50-table historical WTA reference**, opt in explicitly:

```bash
python scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1 --historical-wta --output-dir analysis/slide-mp/wiki10/historical-reference
```

`--historical-wta` cannot be combined with WTA log/configuration options or `--wta-evaluation`. The stored values remain labeled historical; no repetition count, standard deviation, or matched test-sample count is invented for them. Neither mode runs training or loads dataset records.

### Provided Wiki10 WTA logs

One real author-run WTA log, its console output, and its saved CSV are included under `SLIDE/dataset/Wiki10/reference-run/`. They contain a single 30-epoch Wiki10 run and are separate from the default paths used for fresh WTA execution. With a real MP-WTA run already available, use the following command from the artifact root; replace `wiki10-repeat1` with your actual MP-WTA run name:

```bash
python scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1 --wta-log SLIDE/dataset/Wiki10/reference-run/training_wta.log --wta-config SLIDE/dataset/Wiki10/reference-run/Config_wiki10_wta.csv --output-dir analysis/slide-mp/wiki10/provided-wta-comparison
```

This reads the supplied WTA measurements; it does not independently rerun WTA. No real MP-WTA logs or synthetic comparison figures are bundled. The saved WTA CSV records the original training settings; its relative data/output paths are provenance and are not used by the analyzer to launch training. For a fresh WTA run, use the executable and configuration paths in the [SLIDE guide](../SLIDE/README.md).

### Accuracy outputs

**Output directory:** `analysis/slide-mp/<dataset>/`.

| Output | Contents |
| --- | --- |
| `epochs.csv` | MP-WTA per-run, per-epoch P@1 and source lines |
| `summary.csv` | MP-WTA mean P@1, sample standard deviation, and contributing run counts |
| `wta_epochs.csv` | WTA per-run, per-epoch measurements, source lines, and evaluation scope |
| `wta_summary.csv` | WTA mean P@1, sample standard deviation, run counts, and evaluation scope |
| `comparison.csv` | Both methods' epoch statistics, evaluated sample counts, and differences where coverage matches |
| `wta_reference.csv` | Historical WTA values in `--historical-wta` mode; header only in measured-log mode |
| `analysis.json` | Both methods' selections, configuration/log hashes, parsing audit, and baseline provenance |
| `<dataset>_epoch_accuracy.pdf`, `.png` | Measured accuracy curves and available sample SD, or the explicitly selected historical reference |

## Analyzing multiple experiments

Run these commands from the **artifact root**, with the shared Python environment activated. The examples assume the named experiments have actually run and produced usable logs; replace the names with your own. Passing launcher `--check` or completing initialization alone does not provide analysis data. Runtime analysis needs complete batches and rehash calibration; ORAM also needs the next batch-start boundary. Accuracy analysis needs confirmed complete epochs.

| Experiments | How to analyze |
| --- | --- |
| Different datasets | Invoke the analyzer separately for each dataset |
| Different configurations of one dataset | Invoke it separately for each configuration; use separate output directories |
| Independent repetitions of the same configuration | Repeat `--run` in one invocation to calculate a mean |
| Individual runs and their mean | Save each selection with a different `--output-dir` |

Without `--run`, the analyzer discovers usable runs for the requested dataset and applies any parameter filters. It averages the remaining runs only if they form one configuration; otherwise it asks for a more specific selection. This may include `repeat0`; it does not select only the latest run. Epoch limits are ignored for grouping by default, so use exact `--run` names to separate short and full experiments, or add `--param Epoch=30` to filter the recorded limit.

**Separate datasets and configurations.** For example, analyze Amazon-Sub batch32, Wiki325 batch32, and Amazon-Sub batch64 independently:

```bash
python scripts/analyze_tennor.py --dataset amzprime --threads 44 --batchsize 32 --run amzprime-threads44-batchsize32-repeat1 --output-dir analysis/tennor/amzprime/bs32/threads44/repeat1
python scripts/analyze_tennor.py --dataset wiki325 --threads 44 --batchsize 32 --run wiki325-threads44-batchsize32-repeat1 --output-dir analysis/tennor/wiki325/bs32/threads44/repeat1
python scripts/analyze_tennor.py --dataset amzprime --threads 44 --batchsize 64 --run amzprime-threads44-batchsize64-repeat1 --output-dir analysis/tennor/amzprime/bs64/threads44/repeat1
```

The same selection pattern applies to SLIDE-ORAM and MP-WTA using their entry points and options above. MP-WTA commands do not specify threads. Each invocation produces one configuration's analysis; different configurations are not combined into one accuracy curve or averaged together.

**Average independent repetitions.** After collecting a second run with matching settings, select both explicitly:

```bash
python scripts/analyze_tennor.py --dataset amzprime --threads 44 --batchsize 32 --run amzprime-threads44-batchsize32-repeat1 --run amzprime-threads44-batchsize32-repeat2 --output-dir analysis/tennor/amzprime/bs32/threads44/mean-r1-r2
python scripts/analyze_slide_oram.py --dataset amzprime --batchsize 32 --run amzprime-threads1-batchsize32-repeat1 --run amzprime-threads1-batchsize32-repeat2 --output-dir analysis/slide-oram/amzprime/bs32/threads1/mean-r1-r2
python scripts/analyze_slide_mp.py --dataset amzprime --run amzprime-repeat1 --run amzprime-repeat2 --output-dir analysis/slide-mp/amzprime/mean-r1-r2
```

Use distinct repeat IDs and collect repetitions on the same machine with the same execution environment. Copies of a log are not independent repetitions. New MP-WTA runs with an unrecorded thread count form a different group from older runs with a recorded numeric count; an unknown count does not establish that the execution environments match.

Runtime analysis first averages within each run, then gives each contributing run equal weight; accuracy analysis gives each contributing run equal weight at each epoch. Inspect the reported counts, since usable batch or epoch counts can differ. Runtime `--max-batches N` limits the analysis window **per run**; omitted here, it uses all complete batches.

**Preserve individual and aggregate results.** To analyze only the second repetition, retain its `--run` alone and change the destination to a sibling `repeat2/` directory. Rerunning analysis with the same destination replaces the previous analysis there, while raw run logs remain unchanged. MP-WTA's default output directory is shared by all configurations of a dataset, so custom configurations also need distinct `--output-dir` values.

**Select results for the combined runtime table.** To compare the two averaged runtime results above:

```bash
python scripts/build_runtime_table.py --result analysis/tennor/amzprime/bs32/threads44/mean-r1-r2/result.json --result analysis/slide-oram/amzprime/bs32/threads1/mean-r1-r2/result.json --output-dir analysis/runtime/mean-r1-r2
```

This writes the batch32 table under `analysis/runtime/mean-r1-r2/bs32/`. Add further `--result` arguments to include other datasets or TENNOR thread counts. Select one intended result per dataset/batch/method/thread setting; do not include both individual results and their mean for the same setting. The builder checks compatibility before calculating speedups, as described below.

## Combined runtime table

Run this after generating TENNOR and SLIDE-ORAM analyses. **Input:** paired `result.json` and `table_row.csv` files discovered under `analysis/tennor/` and `analysis/slide-oram/`. The builder validates their agreement, units, epoch calculation, and sample provenance.

```bash
python scripts/build_runtime_table.py --list
python scripts/build_runtime_table.py --dataset wiki10 --batchsize 32
```

Omit dataset/batch filters to include all available results, or repeat them to select several values. If multiple candidates exist for a method/thread setting, select the intended results explicitly:

```bash
python scripts/build_runtime_table.py --result analysis/tennor/wiki10/bs32/threads8/result.json --result analysis/slide-oram/wiki10/bs32/threads1/result.json
```

Repeated `--result` options **replace default discovery**, so include both methods and every desired thread setting. Each argument may name `result.json`, its matching `table_row.csv`, or their containing directory. Multiple candidates are not automatically averaged or arbitrarily selected.

Cross-method speedup requires the same dataset and matching known values for `Batchsize`, `InputDim`, `numLayer`, `sizesOfLayers`, `totRecords`, `totRecordsTest`, `Lr`, `Sparsity`, `Rehash`, and `Rebuild`. Method-specific `K`, `L`, and `RangePow` settings may differ across methods. Thread counts remain separate rows; within a method, the algorithm settings must also match for a thread-scaling comparison. Missing shared settings or a mismatch yield no cross-method speedup.

**Speedup = SLIDE-ORAM batch seconds / TENNOR batch seconds.** The displayed units remain hours for ORAM and seconds for TENNOR; the ratio always uses seconds. Each row retains its own run, batch, and input counts.

**Output directory:** `analysis/runtime/bs<size>/`. The builder writes `runtime_table.md`, `.csv`, and `.tex`, corresponding per-dataset tables in subdirectories, and a top-level `result.json` with comparison provenance. To preserve another selection, use `--output-dir analysis/runtime/review-a`; `bs<size>/` is appended automatically.
