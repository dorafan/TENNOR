# TENNOR: build, run, and analyze

The paper evaluates TENNOR on a **Google Cloud C3 server with Intel TDX, 44 physical cores / 88 hardware threads, and 352 GB RAM**.

**TENNOR can also run without TDX, but correct execution requires an Intel x86-64 CPU.** The default build requires AVX2/FMA, and the installation scripts target Ubuntu 24.04. On other Intel x86-64 machines, use the direct commands below; the paper's NUMA/OpenMP prefix is provided at the end.

TENNOR requires substantial memory. Treat the following paper measurements as **minimum available-memory requirements** for the corresponding configurations, and allow additional RAM for the operating system. Insufficient available memory causes allocation failures or forced process termination.

| Dataset | Labels | Batch size 32 | Batch size 64 |
| --- | ---: | ---: | ---: |
| Wiki10-31K | 30,938 | 34.6 GB | 69.5 GB |
| Amazon-Sub | 92,814 | 77.7 GB | 326.3 GB |
| Wiki-325K | 325,056 | 222.8 GB | — |

No batch-64 configuration is supplied for Wiki-325K. Reducing the number of training epochs does not remove the memory needed to initialize the model and its data structures.

Default configurations are in `SLIDE/Config_<dataset>_<threads>_batch<size>.csv`: `wiki10` and `amzprime` have batch-32/64 presets; `wiki325` has batch-32 presets. Each has 8-, 22-, and 44-thread configurations.

## 1. Install dependencies and activate Python

Start at the **artifact root**, the parent of `TENNOR/`:

```bash
bash install_dependencies.sh
source .venv-ae/bin/activate
cd TENNOR
```

The installer installs shared system dependencies, including GCC 13, CMake, unzip, and numactl, and creates `.venv-ae/` for downloads and analysis. Internet access and `sudo` (or root) are required.

**Activate the environment before following the commands below.** Activation lasts for the current shell; repeat it in each new terminal. From `TENNOR/`, use `source ../.venv-ae/bin/activate`. Installation is shared by all components and normally runs only once.

## 2. Download data and build

Run these commands from **`TENNOR/`** with the environment activated:

```bash
bash download_datasets.sh
bash build.sh
```

- **`download_datasets.sh`** downloads Wiki10-31K, WikiLSHTC-325K, and Amazon-670K into `dataset/{Wiki10,Wiki325,Amazon}/`. It prepares Amazon-Sub in `dataset/AmzPrime/` from the bundled `../dataset-Amazon-Sub/AmzPrime/{train.zip,test.txt}`. All datasets become `train.txt` / `test.txt` pairs. Existing nonempty pairs are reused; incomplete pairs cause an error. Downloads use the shared `gdown`.
- **`build.sh`** configures a CMake **Debug** build and builds `bin/runme` from the C++ sources in `SLIDE/`, using four build jobs. CMake fetches and builds zlib and CNPY locally under `bin/`; first builds need GitHub access. OpenMP and AVX2/FMA are enabled by default on x86. After an interrupted or failed build, rerunning the script removes `bin/` and rebuilds it.

## 3. Run TENNOR

**Run all commands in this section from `TENNOR/`, with the shared environment activated.** The launcher manages the executable's working directory automatically.

`run_with_set_tennor.py` reads a configuration from `SLIDE/`, validates parameters and data headers, applies overrides, and launches the already-built `bin/runme`. Configurations contain `KEY=VALUE` lines. Relative input paths such as `../dataset/Wiki10/train.txt` resolve from `TENNOR/bin/`.

List configurations and check an 8-thread Wiki10 run without starting training:

```bash
python run_with_set_tennor.py --list
python run_with_set_tennor.py Config_wiki10_8_batch32.csv --threads 8 --run-name wiki10-threads8-batchsize32-repeat1 --check
```

`--check` requires prepared data and `bin/runme`; it previews the effective configuration without writing files or validating scientific results.

Start training directly on an Intel x86-64 machine; no TDX or `numactl` prefix is required:

```bash
python run_with_set_tennor.py Config_wiki10_8_batch32.csv --threads 8 --run-name wiki10-threads8-batchsize32-repeat1
```

Choose the matching configuration and `--threads` value for another dataset/batch/thread combination. The configuration's thread parameter is spelled **`threatNum`**. `--threads` sets `OMP_NUM_THREADS` but does not change `threatNum`; keep them consistent. For custom settings, repeat `--set KEY=VALUE`, for example `--set Epoch=1` or `--set threatNum=22`. Original configurations are preserved.

Each run gets a new `dataset/runs/<run-name>/` directory:

| File/directory | Contents |
| --- | --- |
| `console.log` | Captured stdout/stderr; input to runtime and breakdown analysis |
| `training.log` | C++ `logFile` output |
| `savedWeights.npz` | Checkpoint, overwritten at each completed epoch |
| `run.json` | Effective parameters, inputs, hashes, OpenMP settings, status, and exit code |
| `work/`, `dataset/` | Working directory and diagnostics, including `dataset/binmaps/` |

The launcher saves `SLIDE/Config_<...>__run_<run-name>.csv` and redirects output paths into the run directory. Existing run names are rejected; use explicit names ending in `repeatN`, as in the examples, for analysis. Keep `isHopper=0` for local diagnostics. Ctrl+C stops training and records an interrupted run.

## 4. Analyze runtime and breakdown

Run these commands from **`TENNOR/`**, with the shared environment activated. After training, [`../scripts/analyze_tennor.py`](../scripts/analyze_tennor.py) reads `dataset/runs/<run-name>/console.log` and its matching `run.json`. It produces amortized mean batch time, estimated epoch time, and Requester/Fetcher/Updater and internal Fetcher breakdowns without rerunning training or loading datasets.

```bash
python ../scripts/analyze_tennor.py --dataset wiki10 --list
python ../scripts/analyze_tennor.py --dataset wiki10 --threads 8 --batchsize 32 --run wiki10-threads8-batchsize32-repeat1
```

Use `--dataset amazonsub` or `--dataset wiki325` for the other datasets. Repeat `--run` to average independent runs with matching parameters. Filters and `--skip-batches` / `--max-batches` select recorded data; they do not change or stop training.

See [Analyzing multiple experiments](../scripts/README.md#analyzing-multiple-experiments) for separate configurations, independent repetitions, output preservation, and comparison-table selection. Its commands run from the artifact root.

Outputs are written to `../analysis/tennor/<dataset>/bs<size>/threads<count>/`:

| Output | Contents |
| --- | --- |
| `table_row.csv`, `table_row.md` | Batch time, estimated epoch time, and sample counts |
| `batches.csv`, `run_summary.csv` | Batch-level and per-run measurements |
| `result.json` | Parameters, provenance, timing details, and analysis diagnostics |
| `<dataset>_bs<size>_threads<count>_breakdown.pdf` / `.png` | Two-panel breakdown figure |

Only complete batches contribute; initialization and unfinished tails are excluded. Batch time includes amortized rehash cost. Logs must contain a complete timed rehash event, or supply a separately measured event cost with `--rehash-seconds`. Epoch time is **extrapolated**, not a measured full epoch. Breakdown panels exclude rehash and residual time, so their sum may differ from the table total; missing component timings prevent figure generation.

Rerunning analysis replaces existing outputs. Use `--output-dir analysis/tennor/<unique-name>` to preserve another selection. Relative analysis path options resolve from the **artifact root**. See the [analysis reference](../scripts/README.md#tennor-runtime-and-breakdown) for details.

After also analyzing a matching SLIDE-ORAM run, generate the combined runtime table and speedups:

```bash
python ../scripts/build_runtime_table.py --dataset wiki10 --batchsize 32
```

This reads both components' analysis results and writes Markdown, CSV, and LaTeX tables under `../analysis/runtime/bs32/`. See the [table reference](../scripts/README.md#combined-runtime-table) for explicit result selection when multiple candidates exist.

## 5. Paper platform: Google Cloud C3 with TDX

For the paper's 44-thread setting, run the following from **`TENNOR/`** with the shared environment activated:

```bash
numactl --cpunodebind=0 --interleave=all env OMP_NUM_THREADS=44 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=threads python run_with_set_tennor.py Config_wiki10_44_batch32.csv --threads 44 --run-name wiki10-threads44-batchsize32-repeat1
```

This binds execution to NUMA node 0, interleaves memory across nodes, and fixes OpenMP thread placement. For 8 or 22 threads, change the configuration, `OMP_NUM_THREADS`, `--threads`, and run name together. Check the topology with `numactl --hardware` before applying these settings to another machine. For analysis, select the corresponding run name and thread count. Runs on other hardware or without TDX can exercise TENNOR, but their timings are not directly comparable to the paper's measurements.
