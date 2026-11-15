# SLIDE: Original WTA Baseline Guide

**For a complete Wiki10 run, follow Sections 2–5 in order. The same build supports the four dataset configurations listed below.**

## 1. Overview and evaluation scope

This component provides an independent WTA baseline using [upstream SLIDE](https://github.com/keroro824/HashingDeepLearning), the implementation accompanying [SLIDE: In Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems](https://arxiv.org/abs/1903.03129). The source is based on commit [`2d10d46b5f6f1eda5d19f27038a596446fc17cee`](https://github.com/keroro824/HashingDeepLearning/tree/2d10d46b5f6f1eda5d19f27038a596446fc17cee), before the HugePages optimization, with a local change to evaluate all complete test batches at every epoch end. HugePages setup is therefore unnecessary for this version.

The added `Config_*_wta.csv` files use **50 output-layer hash tables** (`L=1,50`) and follow the corresponding SLIDE-MP configurations for shared model and training parameters. The dense hidden layer uses one table. The upstream `SLIDE/Config.h` selects densified WTA (`HashFunction=2`), `BUCKETSIZE=128`, `binsize=8`, and training from scratch (`LOADWEIGHT=0`). These are compile-time settings: rebuild after changing them. They are not CSV overrides; in particular, SLIDE-MP uses different bucket-size settings.

This guide covers fresh WTA execution, collection of raw logs, and comparison with MP-WTA. The [accuracy analyzer](../scripts/README.md#mp-wta-accuracy) reads these WTA logs by default; the embedded historical curve is available only through explicit `--historical-wta` selection. See Section 6 for the evaluation schedule and Section 7 for comparison commands.

## 2. Requirements and directory layout

The commands target **Ubuntu 24.04 x86-64**, with Bash, GCC 13, CMake, OpenMP, CNPY, and zlib. Internet access is needed for dependencies and datasets; package installation requires `sudo` or root. This SLIDE build does not require a TEE. Memory and runtime depend on the dataset and machine; Wiki10 is the smallest supplied example. No fixed RAM or completion-time guarantee is provided for the larger datasets.

Keep this component inside the artifact so that the shared installer and bundled Amazon-Sub data are available. Its download and build commands follow the SLIDE-MP workflow:

```text
<artifact-root>/
|-- install_dependencies.sh
|-- .venv-ae/                    Shared Python environment, created by installer
|-- dataset-Amazon-Sub/AmzPrime/ Bundled train.zip and test.txt
|-- SLIDE-MP/                    Existing datasets can be reused automatically
`-- SLIDE/                       Component root: directory containing this README
    |-- README.md
    |-- CMakeLists.txt           Source and dependency build configuration
    |-- build.sh                Builds bin/runme and dependencies
    |-- download_datasets.sh    Prepares all four datasets
    |-- bin/                    Executable, build files, CNPY, and zlib
    |-- dataset/<dataset>/       Input files or links, WTA logs, and checkpoints
    `-- SLIDE/                   Upstream C++ source, Config.h, and CSVs
```

**Start at the artifact root**, then enter this component:

```bash
cd SLIDE
```

**All subsequent command blocks start and end in this outer `SLIDE/` directory**, with the shared environment active after installation. Training runs inside the inner `SLIDE/` directory because CSV paths are resolved relative to the process's working directory.

Install the shared dependencies once:

```bash
bash ../install_dependencies.sh
source ../.venv-ae/bin/activate
```

If the shared installer has already completed for SLIDE-MP or another component, reuse that environment. In a new terminal, return to this component root and activate `../.venv-ae/bin/activate` again. CMake builds CNPY and zlib automatically; no separate installation of either library is needed.

## 3. Download and prepare datasets

Use the same preparation command as SLIDE-MP:

```bash
bash download_datasets.sh
```

The script uses the same data sources and extraction steps as SLIDE-MP. It prepares Wiki10-31K, Wiki-325K, Amazon-670K, and the bundled Amazon-Sub subset, even for the Wiki10 example. Existing nonempty local train/test pairs are reused. If a dataset is already prepared under `../SLIDE-MP/dataset/`, it links only those input files automatically; otherwise it downloads or extracts them into this component's `dataset/` directory. WTA logs and checkpoints remain separate from MP-WTA outputs.

When input files are linked, keep the sibling `SLIDE-MP/dataset/` directory available. Moving the whole artifact together preserves these relative links. Existing incomplete input pairs are reported rather than overwritten.

| Dataset | Local data directory | WTA configuration | Batch size | Input features | Output labels |
| --- | --- | --- | ---: | ---: | ---: |
| Wiki10-31K | `dataset/Wiki10/` | `Config_wiki10_wta.csv` | 32 | 101,938 | 30,938 |
| Amazon-Sub | `dataset/AmzPrime/` | `Config_amzprime_wta.csv` | 128 | 135,908 | 92,814 |
| Wiki-325K | `dataset/Wiki325/` | `Config_wiki325_wta.csv` | 128 | 1,617,899 | 325,056 |
| Amazon-670K | `dataset/Amazon/` | `Config_amz_wta.csv` | 128 | 135,909 | 670,091 |

<details>
<summary>Optional: check dataset headers against the CSV files</summary>

This reads only the configuration files and dataset headers; it does not initialize a model or start training:

```bash
python - <<'PY'
from pathlib import Path

source = Path("SLIDE")
for name in ("wiki10", "amzprime", "wiki325", "amz"):
    csv = source / f"Config_{name}_wta.csv"
    config = {}
    for line in csv.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    features = int(config["InputDim"])
    labels = int(config["sizesOfLayers"].split(",")[-1])
    for key, count in (("trainData", "totRecords"), ("testData", "totRecordsTest")):
        path = source / config[key]
        with path.open() as stream:
            actual = tuple(map(int, stream.readline().split()))
        expected = (int(config[count]), features, labels)
        if actual != expected:
            raise SystemExit(f"{csv}: {path}: expected {expected}, found {actual}")
        print(f"OK: {path}: {actual}")
PY
```

This header check does not scan all records or validate feature indices and checkpoint compatibility.

</details>

Amazon-Sub intentionally has `InputDim=135908`; Amazon-670K has `InputDim=135909`.

## 4. Build

Use the same CMake build command as SLIDE-MP:

```bash
bash build.sh
```

The script configures `bin/`, builds the executable as **`bin/runme`**, and builds zlib and CNPY locally through CMake's `ExternalProject` mechanism. First builds need GitHub access. Dependencies use Release builds; SLIDE uses the historical `-O2`, OpenMP, and `-fno-strict-aliasing` options, even though the CMake configuration is Debug. SLIDE-MP's `-O0` setting is not applied to this upstream baseline. Compatibility options for the older source are already included in CMake.

This uses the same dependency sources as SLIDE-MP: zlib `v1.2.11` and `sarthakpati/cnpy`. CNPY defaults to `HEAD`; to pin a revision, pass `-DCNPY_GIT_COMMIT=<commit>` to `bash build.sh`. Record the resolved CNPY revision with your results. No Makefile editing, manual compiler command, or library-path setup is required.

Rerun `bash build.sh` after changing C++ source or `SLIDE/Config.h`. Changing only the dataset CSV does not require rebuilding. The executable accepts a CSV filename; it does not implement the other components' `--check`, `--set`, or `--build-info` options. The optional header check in Section 3 can validate dataset metadata before execution.

## 5. Run WTA training

After dataset preparation and the build succeed, run the complete **30-epoch Wiki10 example** below from this component root. `tee` displays the console output and saves it to a file. The final `cd ..` returns to this component root.

The included author-run logs are preserved separately in `dataset/Wiki10/reference-run/`; this command writes fresh outputs in `dataset/Wiki10/` and leaves those supporting logs intact. For subsequent runs, preserve your previous outputs as described below.

```bash
cd SLIDE
../bin/runme Config_wiki10_wta.csv 2>&1 | tee ../dataset/Wiki10/console_wta.log
cd ..
```

For another dataset, replace the CSV filename and console-log path in the same command:

| Dataset | CSV filename | Console-log path |
| --- | --- | --- |
| Amazon-Sub | `Config_amzprime_wta.csv` | `../dataset/AmzPrime/console_wta.log` |
| Wiki-325K | `Config_wiki325_wta.csv` | `../dataset/Wiki325/console_wta.log` |
| Amazon-670K | `Config_amz_wta.csv` | `../dataset/Amazon/console_wta.log` |

Use the supplied `Config_*_wta.csv` files rather than the upstream `Config_amz.csv`, which retains its original paths and settings. Training uses the OpenMP runtime's thread selection and honors inherited OpenMP environment variables. Record the thread settings, hardware, and compiler when comparing runs.

<details>
<summary>Optional: preserve outputs for repeated runs</summary>

Before repeating a run, check for existing `training_wta.log`, `savedWeights_wta.npz`, and `console_wta.log` in that dataset's directory. The program appends to the training log and overwrites the checkpoint; `tee` overwrites the console log.

To preserve earlier results, archive all three files first. Alternatively, copy the CSV and set fresh `logFile`, `savedweight`, and `weight` paths, then use that CSV and a new console-log path in the command. Changing only the `tee` filename does not isolate the training log or checkpoint. With the supplied `LOADWEIGHT=0`, each invocation starts training from scratch.

</details>

## 6. Outputs and accuracy interpretation

The package includes one real 30-epoch Wiki10 WTA run in [`dataset/Wiki10/reference-run/`](dataset/Wiki10/reference-run/README.md): `training_wta.log`, `console_wta.log`, and the matching `Config_wiki10_wta.csv`. The training log contains epoch-end measurements for epochs 0 through 29. These are supporting logs from an author run, not an average of independent runs; no checkpoint or MP-WTA run is included. The [analysis guide](../scripts/README.md#provided-wiki10-wta-logs) shows how to select this preserved log explicitly.

For the Wiki10 example, retain:

| Output | Contents |
| --- | --- |
| `dataset/Wiki10/training_wta.log` | Epoch markers and evaluation rows: iteration, accumulated training time in seconds, accuracy |
| `dataset/Wiki10/console_wta.log` | Standard output and error, including initialization and evaluation progress |
| `dataset/Wiki10/savedWeights_wta.npz` | Latest saved model checkpoint |

Other datasets use the corresponding directory from Section 3. Keep the CSV, `SLIDE/Config.h`, source revision, `bin/CMakeCache.txt`, resolved dependency revisions, compiler version, and OpenMP settings with each run. This upstream runner does not create the artifact launchers' `run.json` metadata.

Record the dependency revisions after building, from this component root:

```bash
git -C bin/_deps/cnpy/src/CNPY rev-parse HEAD
git -C bin/_deps/zlib/src/ZLIB rev-parse HEAD
```

The modified `main.cpp` uses two evaluation sizes:

- During training, every `Stepsize` iterations: the first **20 complete test batches**.
- At the end of every epoch, including the final epoch: **all complete test batches**, matching SLIDE-MP's epoch-end coverage. This is `floor(totRecordsTest / Batchsize)` batches; the final incomplete batch is excluded.

For the supplied `Epoch=30` configurations, epoch markers run from 0 to 29. Training also uses complete batches only. Each epoch-end evaluation has iteration `(epoch + 1) * floor(totRecords / Batchsize)`. The intermediate 20-batch measurements remain partial-test evaluations; the analyzer selects epoch-end measurements.

**Recompile SLIDE using Section 4 before collecting new logs with this schedule.** The CSV fields and three-column log format are unchanged. Preserve earlier logs separately and use fresh output files: a log alone does not identify which evaluation schedule generated it. For logs from the unmodified upstream schedule, use the explicit legacy option described in the [analysis guide](../scripts/README.md#mp-wta-accuracy).

The upstream `python_examples/` TensorFlow examples are not used by this WTA workflow.

## 7. Compare with MP-WTA

After collecting both the WTA log above and a matching [MP-WTA run](../SLIDE-MP/README.md), run this from the outer `SLIDE/` directory with the shared Python environment active:

```bash
python ../scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1 --wta-log SLIDE/dataset/Wiki10/training_wta.log --wta-config SLIDE/SLIDE/Config_wiki10_wta.csv --output-dir analysis/slide-mp/wiki10/measured-comparison
```

All explicit analysis paths resolve from the **artifact root**, even though this command runs from `SLIDE/`. The WTA log/configuration options shown match the Wiki10 defaults and may be omitted for an unchanged default run. Other datasets use their directories and CSVs from Section 3. Missing WTA logs cause an error instead of selecting historical values automatically.

Repeat `--wta-log` to aggregate independent WTA repetitions; repeat `--run` independently for MP-WTA. One `--wta-config` can describe all selected WTA runs, or supply one per WTA log in the same order. Use the saved configuration that actually produced each log. The analyzer requires matching WTA configurations and rejects duplicated WTA paths or identical log copies. See the [full analysis examples](../scripts/README.md#mp-wta-accuracy).

The analyzer uses the shared `Epoch`/three-column log format for both methods. For WTA, an intact epoch-end record at `(epoch + 1) * floor(totRecords / Batchsize)` confirms the evaluation, including the final epoch without `run.json`. It does not certify that checkpoint saving or the training process finished successfully. Keep interrupted or malformed records in the raw log; the analysis audit records which blocks were excluded.

Inspect `wta_epochs.csv`, `wta_summary.csv`, `comparison.csv`, and the PDF/PNG in `../analysis/slide-mp/wiki10/measured-comparison/`. The original `epochs.csv` and `summary.csv` contain MP-WTA measurements. Both methods' source hashes, selections, configuration evidence, and parsing audits appear in `analysis.json`.

New WTA logs use all complete test batches at every epoch end, as MP-WTA does. The analyzer defaults to this schedule (`--wta-evaluation full`), and the comparison CSV reports **MP-WTA minus WTA** when both evaluated sample counts are known and equal. Keep the datasets and batch sizes aligned. No missing epochs are filled, and sample standard deviation is reported only when at least two independent logs contribute.
