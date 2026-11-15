# TENNOR: Artifact Evaluation Guide

**For an end-to-end Wiki10 walkthrough, from dependency installation to all analysis steps, see [Appendix A](#appendix-a-end-to-end-wiki10-walkthrough-and-minimal-check). It includes optional early stopping for TENNOR and SLIDE-ORAM, and a complete MP-WTA run compared with the historical WTA reference. For a fresh WTA comparison, also follow the [SLIDE guide](SLIDE/README.md).**

## 1. Overview and evaluation scope

This artifact accompanies [TENNOR: Trustworthy Execution for Neural Networks through Obliviousness and Retrievals](https://arxiv.org/abs/2605.07160v2). It provides TENNOR, the SLIDE-ORAM performance baseline (called SLIDE+ORAM in the paper), SLIDE-MP for MP-WTA accuracy evaluation, and upstream SLIDE for fresh WTA baseline runs, with dataset preparation, build/run instructions, and result analysis.

The evaluation covers three main results. Paper references below use arXiv version 2.

| Evaluated result | Paper reference | Execution guide | Analysis guide |
| --- | --- | --- | --- |
| TENNOR runtime and breakdown | Table 2 and Figure 2 | [TENNOR](TENNOR/README.md) | [Runtime and breakdown](scripts/README.md#tennor-runtime-and-breakdown) |
| SLIDE-ORAM runtime | Table 2 | [SLIDE-ORAM](SLIDE-ORAM/README.md) | [Baseline runtime](scripts/README.md#slide-oram-runtime) |
| MP-WTA accuracy versus original WTA | Section 7.3 and Figure 7 | [SLIDE-MP](SLIDE-MP/README.md) and [SLIDE WTA](SLIDE/README.md) | [Accuracy](scripts/README.md#mp-wta-accuracy) |

The first two analyses also produce a combined runtime and speedup table. Accuracy analysis reads fresh 50-table WTA logs by default; the earlier fixed WTA reference remains available through explicit `--historical-wta` selection. Missing WTA logs do not trigger an automatic historical fallback. Ablations and the paper's remaining experiments are outside this evaluation workflow.

## 2. Requirements and expected cost

**The paper's TENNOR and SLIDE-ORAM experiments used Google Cloud C3 with Intel TDX, 44 physical cores / 88 hardware threads, and 352 GB RAM.** Accuracy experiments used a bare-metal 64-core AMD EPYC 7763 (Zen 3) server without a TEE. See [Section 7](https://arxiv.org/pdf/2605.07160v2#page=9).

| Component | Supported execution environment | Memory and parallelism |
| --- | --- | --- |
| TENNOR | Intel x86-64 with AVX2/FMA; TDX is optional for running the code | Substantial RAM required; supplied presets use 8, 22, or 44 threads |
| SLIDE-ORAM | Ordinary x86-64; TDX is optional for running the code | 32 GB RAM recommended for the default Wiki10 build; larger capacity presets require more RAM; one effective thread |
| SLIDE-MP | AMD or Intel x86-64; no TDX requirement | Memory depends on dataset/model size; OpenMP chooses the thread count |
| SLIDE WTA | Ordinary x86-64; no TEE or HugePages requirement for the selected upstream revision | Memory depends on dataset/model size; OpenMP honors the caller's environment |

For TENNOR at batch size 32, allow at least **34.6 GB for Wiki10-31K, 77.7 GB for Amazon-Sub, or 222.8 GB for Wiki-325K**, plus memory for the operating system. Insufficient available memory can terminate the process. Larger batches require more memory; consult the [complete memory table](TENNOR/README.md) before selecting a configuration.

**Different machines and configurations may change the final results.** CPU architecture, memory capacity and bandwidth, thread count, NUMA placement, compiler settings, and TDX can affect runtime and its breakdown. Training parameters and randomness can also affect accuracy. Use the documented experimental environment and configurations for comparison with the paper, and report any differences. Runtime measurements vary across hardware platforms and configurations.

The installation scripts require **Ubuntu 24.04 x86-64**, internet access, and `sudo` or root privileges. They install the GCC 13/CMake toolchain and a shared Python environment. First builds fetch dependencies from GitHub; dataset preparation downloads and extracts data separately for each component. Reserve space for archives, extracted datasets, builds, logs, and checkpoints.

**SLIDE-ORAM is very slow, including at `Epoch=1`.** As a planning reference, Table 2 reports approximately 2.6 hours per Wiki10 baseline batch: 40 such batches correspond to about 104 hours, before initialization and other overhead. Larger workloads cost substantially more. This is an estimate from the paper, not a measured installation-to-completion budget for this package. Download/build time and MP-WTA training time depend on the target machine and connection. Analysis of existing logs avoids the cost of rerunning training.

## 3. Artifact layout

```text
.
|-- TENNOR/                 TENNOR configurations, build/run tools, and README
|-- SLIDE-ORAM/             ORAM baseline configurations, build/run tools, and README
|-- SLIDE-MP/               MP-WTA configurations, build/run tools, and README
|-- SLIDE/                  Original WTA source, configurations, and execution guide
|-- scripts/                Analysis entry points, shared modules, and README
|-- analysis/               Generated tables, figures, and analysis provenance
|-- install_dependencies.sh Shared dependency installer
`-- README.md               Evaluation entry point
```

Within each component, `SLIDE/` holds the source/configuration tree and `bin/` holds the generated executable. TENNOR, SLIDE-ORAM, and SLIDE-MP keep independent outputs in `dataset/runs/<run-name>/`. Original WTA uses `SLIDE/bin/runme` and writes outputs under `SLIDE/dataset/<dataset>/`; see its guide for preserving repetitions. Its download and CMake build commands follow SLIDE-MP's workflow and can reuse existing MP-WTA input datasets. The download scripts prepare Wiki10-31K, Wiki-325K, and Amazon-670K, and use the root-level `dataset-Amazon-Sub/AmzPrime/` bundle for Amazon-Sub. Default configurations are listed in the component guides. All supported analysis scripts are documented together in [scripts/README.md](scripts/README.md).

## 4. Installation, build, and preflight checks

**Start at the artifact root**, the directory containing this README:

```bash
bash install_dependencies.sh
source .venv-ae/bin/activate
```

Install once for all components. Activate the environment before continuing and again in each new terminal; changing directories does not deactivate it. The installer creates `.venv-ae/` but cannot activate the calling shell. It records installed versions in `.venv-ae/requirements-resolved.txt` and `.venv-ae/system-packages.txt`.

Prepare each component below. **Each block starts and ends at the artifact root**, with the shared environment active. The download scripts prepare all supported datasets, even though these examples select Wiki10 for execution. The build scripts produce each component's `bin/runme`.

### TENNOR

```bash
cd TENNOR
bash download_datasets.sh
bash build.sh
python run_with_set_tennor.py Config_wiki10_44_batch32.csv --threads 44 --run-name wiki10-threads44-batchsize32-repeat1 --check
cd ..
```

### SLIDE-ORAM

```bash
cd SLIDE-ORAM
bash download_datasets.sh
bash build.sh --dataset wiki10
python run_with_set_oram.py Config_wiki10_batch32.csv --threads 1 --run-name wiki10-threads1-batchsize32-repeat1 --check
cd ..
```

### SLIDE-MP

```bash
cd SLIDE-MP
bash download_datasets.sh
bash build.sh
python run_with_set.py Config_wiki10.csv --run-name wiki10-repeat1 --check
cd ..
```

### SLIDE WTA

Prepare the original WTA baseline with the same download and build commands. The downloader reuses prepared SLIDE-MP inputs when available, and CMake builds CNPY and zlib automatically:

```bash
cd SLIDE
bash download_datasets.sh
bash build.sh
cd ..
```

This produces `SLIDE/bin/runme`. Follow the [SLIDE guide](SLIDE/README.md#5-run-wta-training) for execution; its upstream runner accepts a CSV directly and does not support `--check`.

`--check` validates inputs and previews the effective configuration without starting training or creating a run. For SLIDE-ORAM, it also queries the actual `bin/runme` for its compiled settings and checks ORAM block capacity against the effective configuration, including overrides. This short query exits before model initialization; older binaries without query support require rebuilding. Normal SLIDE-ORAM launches repeat the same capacity check before creating run files. See [compiled capacity preflight](SLIDE-ORAM/README.md#compiled-capacity-preflight) for the checks and their limits: `SERIAL_LEN` is reported but not validated, and checkpoint compatibility and scientific results are not checked.

Before switching to Amazon-Sub or Wiki-325K, rebuild with `bash build.sh --dataset amzprime` or `bash build.sh --dataset wiki325` from `SLIDE-ORAM/`; see its [capacity presets and build/run examples](SLIDE-ORAM/README.md#dataset-capacity-presets). All presets build the same `bin/runme`. Section 5 exercises the components. For a shorter execution and analysis check, use [Appendix A](#appendix-a-end-to-end-wiki10-walkthrough-and-minimal-check).

## 5. Reproducing the main results

Complete Section 4 first. **Each command block below starts and ends at the artifact root**, with the shared environment active. Wiki10 is the least expensive evaluation dataset and is used for all examples. Keep TENNOR, SLIDE-ORAM, and SLIDE-MP logs with their original `run.json`; upstream SLIDE WTA instead requires its saved CSV and execution records. Run names must be new; preflight checks do not reserve them.

The component examples run training and analysis from the same component directory. Use `../scripts/` there; the final `cd ..` returns to the artifact root after analysis.

For multiple experiments, analyze each dataset and configuration separately, and combine independent repetitions of the same configuration with repeated `--run` options. Use explicit run names ending in `repeat1`, `repeat2`, etc., and separate `--output-dir` destinations to preserve individual and averaged results. See [Analyzing multiple experiments](scripts/README.md#analyzing-multiple-experiments) for examples, naming requirements, and explicit result selection for comparison tables.

Table 2 reports a window of 40 complete batches at batch size 32, corresponding to 1,280 inputs. The commands below analyze **all complete batches in the selected logs**; report the actual batch counts and measurement window when comparing results. TENNOR and ORAM continue until their configured epoch count is reached or you press Ctrl+C. A usable runtime sample needs complete batches and a complete timed rehash event. ORAM also needs the next consecutive batch-start timestamp to close its last measured batch. See the [measurement rules](scripts/README.md#runtime-outputs-and-measurement-rules) for rehash calibration and partial logs.

### TENNOR runtime and breakdown

Run the Wiki10 example with 44 threads on a supported Intel machine. Let training finish, or stop after collecting the intended measurement window and a complete timed rehash event, then analyze:

```bash
cd TENNOR
python run_with_set_tennor.py Config_wiki10_44_batch32.csv --threads 44 --run-name wiki10-threads44-batchsize32-repeat1
python ../scripts/analyze_tennor.py --dataset wiki10 --threads 44 --batchsize 32 --run wiki10-threads44-batchsize32-repeat1
cd ..
```

Inspect `analysis/tennor/wiki10/bs32/threads44/table_row.md` and the breakdown PDF/PNG in that directory. Repeat for 8/22 threads and Amazon-Sub or Wiki-325K using the corresponding configuration and run name. The [TENNOR guide](TENNOR/README.md) includes those presets, optional batch-64 measurements, and the paper's Google Cloud NUMA/OpenMP command.

### SLIDE-ORAM runtime

Run the matching Wiki10 baseline with one effective thread. Stop after collecting the required measurements and their closing boundary, then analyze:

```bash
cd SLIDE-ORAM
python run_with_set_oram.py Config_wiki10_batch32.csv --threads 1 --run-name wiki10-threads1-batchsize32-repeat1
python ../scripts/analyze_slide_oram.py --dataset wiki10 --batchsize 32 --run wiki10-threads1-batchsize32-repeat1
cd ..
```

Inspect `analysis/slide-oram/wiki10/bs32/threads1/table_row.md`. Use the [SLIDE-ORAM capacity presets](SLIDE-ORAM/README.md#dataset-capacity-presets) before running other datasets; selecting a CSV does not rebuild or resize the executable. Increasing `--threads` does not parallelize training.

### MP-WTA accuracy

First collect a fresh Wiki10 WTA log using the [SLIDE build/run guide](SLIDE/README.md), preserving the CSV used for that run. Then run one complete MP-WTA experiment with the default 30-epoch configuration and analyze both logs. The default WTA inputs are `SLIDE/dataset/Wiki10/training_wta.log` and `SLIDE/SLIDE/Config_wiki10_wta.csv`. SLIDE-MP leaves thread selection to OpenMP and clears inherited thread controls only for its own child process; this does not change the shell environment or the other components.

The package also includes raw logs and a saved configuration from one author-run, 30-epoch Wiki10 WTA experiment in `SLIDE/dataset/Wiki10/reference-run/`. They are separate from fresh-run outputs. To use those supporting logs with your MP-WTA run, follow the [provided-log example](scripts/README.md#provided-wiki10-wta-logs); this analyzes an existing WTA run rather than independently rerunning it.

```bash
cd SLIDE-MP
python run_with_set.py Config_wiki10.csv --run-name wiki10-repeat1
python ../scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1
cd ..
```

To collect five MP-WTA repetitions, rerun the launcher with names ending in `repeat2` through `repeat5`, then select all five from the artifact root:

```bash
python scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1 --run wiki10-repeat2 --run wiki10-repeat3 --run wiki10-repeat4 --run wiki10-repeat5
```

This command still uses one WTA log. To average independent WTA repetitions as well, add one `--wta-log PATH` per preserved log and supply its actual CSV with `--wta-config PATH`; see the [comparison and repetition examples](scripts/README.md#mp-wta-accuracy). The two methods' run counts are reported separately.

Inspect `analysis/slide-mp/wiki10/summary.csv`, `wta_summary.csv`, `comparison.csv`, and `wiki10_epoch_accuracy.pdf` or `.png`. The modified SLIDE WTA and SLIDE-MP both evaluate all complete test batches at every epoch end, excluding the final incomplete batch. Recompile SLIDE before collecting new logs; analysis defaults to this schedule. For older upstream logs, select the legacy schedule described in the [analysis guide](scripts/README.md#mp-wta-accuracy). The comparison records evaluation coverage and emits accuracy differences only when the evaluated sample counts are known and equal. Other canonical dataset names are `wiki325`, `amazon670`, and `amzprime`; see the component guides for their configurations.

For the original stored-reference workflow without collecting a new WTA run, add `--historical-wta` and use a distinct output directory. That mode explicitly labels the historical curve and does not treat it as a newly reproduced result.

### Combined runtime table

After both runtime analyses, run this from the artifact root:

```bash
python scripts/build_runtime_table.py --result analysis/tennor/wiki10/bs32/threads44/result.json --result analysis/slide-oram/wiki10/bs32/threads1/result.json
```

Inspect `analysis/runtime/bs32/runtime_table.md`. Add further `--result` arguments to include other TENNOR thread counts; explicit selection includes only the listed results. See the [table guide](scripts/README.md#combined-runtime-table) for compatibility rules.

To analyze previously collected logs, use each analyzer's `--dataset NAME --list` to discover its available runs, then select the actual folder names with `--run`. This route needs neither a C++ build nor training datasets. It verifies result processing; fresh execution is covered by the component workflows above. Recorded runs retain their original settings even when default configurations have since changed.

## 6. Expected results and interpretation

| Result | Evidence to inspect | Comparison objective |
| --- | --- | --- |
| TENNOR runtime and breakdown | `table_row.md`, breakdown figure, and `result.json` | Lower batch time than the compatible ORAM baseline; thread scaling and the contribution of Requester, Fetcher, and Updater |
| SLIDE-ORAM runtime | Baseline row and combined `runtime_table.md` | Sequential baseline cost and TENNOR/ORAM speedup for the same training task |
| MP-WTA accuracy | `summary.csv`, `wta_summary.csv`, `comparison.csv`, accuracy figure, and `analysis.json` | Compare MP-WTA's single-table accuracy with measured 50-table WTA, accounting for evaluation coverage; the historical reference is an explicit alternative |

Runtime statistics use complete batches, exclude initialization and unfinished tails, and include amortized rehash cost. **Epoch time is extrapolated from mean batch time**, not measured by timing an entire epoch. Breakdown panels exclude rehash and residual time. The combined table computes speedup using seconds for both methods, although ORAM batch time is displayed in hours.

Accuracy uses completed-epoch P@1 on the 0–1 scale. The analyzer reports actual contributing run counts and sample standard deviation when at least two runs contribute; missing epochs are not filled. Inspect counts and provenance alongside the figures. Exact numerical agreement is not guaranteed across machines or random runs. Investigate differences in configuration, data, measurement window, and execution environment before interpreting a discrepancy as an algorithmic result.

## 7. Reusing the artifact

Use launcher `--set KEY=VALUE` options to explore configurations while preserving the supplied defaults. Keep TENNOR's configuration thread count consistent with its OpenMP setting, and rebuild after changing source or compile-time ORAM parameters. Assign a new run name to each experiment.

Analysis filters such as `--param KEY=VALUE` select recorded settings; they do not change training. Use repeated `--run` options for matching independent repetitions and separate `--output-dir` destinations for different selections. Explicit relative analysis paths resolve from the artifact root, even when the command is invoked from a component directory. Preserve `run.json`, raw logs, installed-package records, and analysis JSON files with the source version used. Detailed options and supported output formats are in the component guides and [scripts/README.md](scripts/README.md).

## 8. Availability, license, citation, and contact

The artifact is distributed as source code, configurations, automation scripts, and analysis tools. Its code is released under the [MIT License](LICENSE); third-party code and datasets retain their respective terms, documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

<!-- Publication metadata: add the public repository URL, exact release tag/commit, and permanent artifact archive DOI before submission. The citation below identifies the paper, not the artifact. -->

Please cite the accompanying paper when using this artifact:

```bibtex
@misc{qu2026tennor,
  title = {TENNOR: Trustworthy Execution for Neural Networks through Obliviousness and Retrievals},
  author = {Zifan Qu and Vasileios P. Kemerlis and Giuseppe Ateniese and Evgenios M. Kornaropoulos},
  year = {2026},
  eprint = {2605.07160},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR},
  url = {https://arxiv.org/abs/2605.07160}
}
```

For questions, use the author contact details on the [paper's first page](https://arxiv.org/pdf/2605.07160v2#page=1). Include the component, configuration, command, environment, and relevant logs when reporting a problem.

## Appendix A. End-to-end Wiki10 walkthrough and minimal check

This walkthrough covers preparation, execution, and all four analysis entry points for Wiki10. The examples analyze two complete batches each for TENNOR and SLIDE-ORAM, and a complete 30-epoch run for MP-WTA against the explicitly selected historical WTA reference. It does not rerun WTA; use the [SLIDE guide](SLIDE/README.md) and the fresh-log comparison in Section 5 for that extension. The runtime runs may be stopped early at the markers below or allowed to continue. The full sequence requires an Intel x86-64 machine with AVX2/FMA and sufficient RAM for TENNOR; TDX is optional. **Even this small ORAM run can take many hours**, so MP-WTA comes before SLIDE-ORAM below. Dataset preparation still downloads all supported datasets.

**Two batches are a sampling choice for this walkthrough, not a requirement to run either component.** In A.2 and A.4, `--max-batches 2` selects at most the first two complete batches per run for analysis; it does not stop training. Omit this option to analyze all complete batches in the selected logs. Rehash amortization still requires a complete timed rehash event or a separately measured calibration, as described in the [measurement rules](scripts/README.md#runtime-outputs-and-measurement-rules).

Run the blocks in order, starting at the artifact root with unused `repeat0` run names. If stopping a runtime run early, press Ctrl+C once at the specified marker and wait for the launcher to finish cleanup and return to the shell before continuing. These planned interruptions produce `status=interrupted` in `run.json` and a nonzero launcher exit code; their complete batches remain valid analysis inputs. Run subsequent commands separately, without joining them to the launcher with `&&`. Let MP-WTA finish all 30 configured epochs normally.

In A.2–A.4, analyze from the component directory using `../scripts/`, then return to the root with `cd ..`. The `--output-dir` values still resolve from the artifact root and do not need a `../` prefix.

### A.1 Install, download, build, and check

This block starts and ends at the artifact root. If Section 4 is already complete, reuse that installation, prepared data, and builds; activate the shared environment and proceed to A.2.

```bash
bash install_dependencies.sh
source .venv-ae/bin/activate
cd TENNOR
bash download_datasets.sh
bash build.sh
python run_with_set_tennor.py Config_wiki10_44_batch32.csv --threads 44 --run-name wiki10-threads44-batchsize32-repeat0 --check
cd ..
cd SLIDE-MP
bash download_datasets.sh
bash build.sh
python run_with_set.py Config_wiki10.csv --run-name wiki10-repeat0 --check
cd ..
cd SLIDE-ORAM
bash download_datasets.sh
bash build.sh --dataset wiki10
python run_with_set_oram.py Config_wiki10_batch32.csv --threads 1 --run-name wiki10-threads1-batchsize32-repeat0 --check
cd ..
```

### A.2 Run and analyze TENNOR

From the artifact root, start training. For the minimal check, you may press Ctrl+C **after `Training Batch 1 takes ... seconds` appears**. TENNOR's first rehash is already in batch 0; including batch 1 also checks an ordinary batch. Selecting two batches here exercises both cases and keeps the walkthrough's sample size consistent with A.4.

```bash
cd TENNOR
python run_with_set_tennor.py Config_wiki10_44_batch32.csv --threads 44 --run-name wiki10-threads44-batchsize32-repeat0
```

After the launcher returns, **remain in `TENNOR/` and run the analysis below**, then return to the root:

```bash
python ../scripts/analyze_tennor.py --dataset wiki10 --threads 44 --batchsize 32 --run wiki10-threads44-batchsize32-repeat0 --max-batches 2 --output-dir analysis/tennor/wiki10/initial-check
cd ..
```

### A.3 Run and analyze MP-WTA

From the artifact root, run the default 30-epoch Wiki10 configuration to completion, then plot the full accuracy curve against the stored WTA reference. In the authors' tests, this run took less than one hour; runtime varies across machines. This block finishes at the root.

```bash
cd SLIDE-MP
python run_with_set.py Config_wiki10.csv --run-name wiki10-repeat0
python ../scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat0 --historical-wta --output-dir analysis/slide-mp/wiki10/initial-check
cd ..
```

### A.4 Run and analyze SLIDE-ORAM

**SLIDE-ORAM is single-threaded and very slow: even this two-batch Wiki10 check can take many hours.** The earlier steps let you inspect TENNOR and the full MP-WTA accuracy curve before starting this baseline.

From the artifact root, start the baseline. For the minimal check, you may press Ctrl+C **after `START TRAINING Batch 2` appears**. This closes batches 0 and 1, including the first rehash measurement in batch 1. Stopping at the start of batch 1 or immediately after its rehash would leave that batch incomplete.

```bash
cd SLIDE-ORAM
python run_with_set_oram.py Config_wiki10_batch32.csv --threads 1 --run-name wiki10-threads1-batchsize32-repeat0
```

After the launcher returns, **remain in `SLIDE-ORAM/` and run the analysis below**, then return to the root:

```bash
python ../scripts/analyze_slide_oram.py --dataset wiki10 --batchsize 32 --run wiki10-threads1-batchsize32-repeat0 --max-batches 2 --output-dir analysis/slide-oram/wiki10/initial-check
cd ..
```

### A.5 Build the comparison table and inspect outputs

From the artifact root, combine only the two runtime results from this walkthrough. The expected batch counts below assume the commands are used as shown, with `--max-batches 2`:

```bash
python scripts/build_runtime_table.py --result analysis/tennor/wiki10/initial-check/result.json --result analysis/slide-oram/wiki10/initial-check/result.json --output-dir analysis/runtime/initial-check
```

| Check | Expected output |
| --- | --- |
| TENNOR | `analysis/tennor/wiki10/initial-check/`: timing row, `result.json`, and breakdown PDF/PNG; two selected batches |
| MP-WTA | `analysis/slide-mp/wiki10/initial-check/`: `summary.csv` containing epochs 0–29, `analysis.json`, and the full accuracy PDF/PNG; one contributing run |
| SLIDE-ORAM | `analysis/slide-oram/wiki10/initial-check/`: timing row and `result.json`; two selected batches and one effective thread |
| Combined table | `analysis/runtime/initial-check/bs32/runtime_table.md`, `.csv`, `.tex`, and `result.json` |

Check the selected counts and diagnostics in the JSON outputs. The MP-WTA run should have `status=completed` and `returncode=0` in its `run.json`; runtime runs stopped with Ctrl+C should be marked interrupted. `--max-batches 2` limits analysis only and does not stop training or create missing measurements. Keep the two-batch runtime samples separate from the larger measurement windows used to evaluate performance. The MP-WTA result is a complete single-run curve; the paper averages five independent runs.
