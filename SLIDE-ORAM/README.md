# SLIDE-ORAM: build, run, and analyze

The paper evaluates SLIDE-ORAM on the same **Google Cloud C3 server with Intel TDX, 44 physical cores / 88 hardware threads, and 352 GB RAM** used for TENNOR.

**SLIDE-ORAM also runs on ordinary x86-64 machines without TDX. We recommend 32 GB RAM for the default Wiki10 build**; larger capacity presets require more memory and initialization time. Actual memory use depends on the dataset and compiled ORAM size. The installation scripts target Ubuntu 24.04.

**Training is not parallelized: SLIDE-ORAM runs with one effective thread and is very slow.** Increasing `--threads` does not enable parallel training. Even the default `Epoch=1` run can take substantial time. Use the direct commands below on ordinary machines.

Default configurations are in `SLIDE/Config_<dataset>_batch32.csv`, for `wiki10`, `wiki325`, `amzprime`, and `amz`. All use `L=1,50`, `Batchsize=32`, and `Epoch=1`.

## 1. Install dependencies and activate Python

Start at the **artifact root**, the parent of `SLIDE-ORAM/`:

```bash
bash install_dependencies.sh
source .venv-ae/bin/activate
cd SLIDE-ORAM
```

The installer installs shared system dependencies, including GCC 13, CMake, unzip, and numactl, and creates `.venv-ae/` for downloads and analysis. Internet access and `sudo` (or root) are required. The launcher also requires GNU `stdbuf`, supplied by Ubuntu's `coreutils`.

**Activate the environment before following the commands below.** Activation lasts for the current shell; repeat it in each new terminal. From `SLIDE-ORAM/`, use `source ../.venv-ae/bin/activate`. If dependencies were installed for another component, reuse the same installation.

## 2. Download data and build

Run these commands from **`SLIDE-ORAM/`** with the environment activated:

```bash
bash download_datasets.sh
bash build.sh
```

- **`download_datasets.sh`** downloads Wiki10-31K, WikiLSHTC-325K, and Amazon-670K into `dataset/{Wiki10,Wiki325,Amazon}/`. It prepares Amazon-Sub in `dataset/AmzPrime/` from the bundled `../dataset-Amazon-Sub/AmzPrime/{train.zip,test.txt}`. All datasets become `train.txt` / `test.txt` pairs. Existing nonempty pairs are reused; incomplete pairs cause an error. Downloads use the shared `gdown`.
- **`build.sh`** configures a CMake **Debug** build and builds `bin/runme` from the C++ sources in `SLIDE/`, using four build jobs. CMake fetches and builds zlib and CNPY locally under `bin/`; first builds need GitHub access. After an interrupted or failed build, rerunning the script removes `bin/` and rebuilds it. Parallel build jobs do not make training parallel.

**Before changing datasets, select the matching [capacity preset](#dataset-capacity-presets) and rebuild.** `ORAM_N` and `SERIAL_LEN` are compile-time settings: changing a CSV cannot resize the structures. The presets set `ORAM_N` through the build and leave `SERIAL_LEN=6144` unchanged for the supplied batch-32 configurations. For custom configurations, check both settings for sufficient capacity. Training from scratch uses `LOADWEIGHT=0`; loading an existing checkpoint requires a build with `LOADWEIGHT=1` and a compatible `weight` file.

### Dataset capacity presets

The following presets cover the supplied Wiki10, Amazon-Sub, and Wiki-325K batch-32 configurations. Required logical blocks are the sum of the node count and `L * 2^RangePow` hash buckets for each sparse layer. `ORAM_N` must be a power of two and at least this total.

| Dataset | Runtime configuration | Required logical blocks | Compiled `ORAM_N` | Build command |
| --- | --- | ---: | ---: | --- |
| Wiki10-31K | `Config_wiki10_batch32.csv` | 56,538 | 65,536 | `bash build.sh --dataset wiki10` |
| Amazon-Sub | `Config_amzprime_batch32.csv` | 297,614 | 524,288 | `bash build.sh --dataset amzprime` |
| Wiki-325K | `Config_wiki325_batch32.csv` | 1,963,456 | 2,097,152 | `bash build.sh --dataset wiki325` |

An explicit preset overrides the `ORAM_N` default in `SLIDE/Config.h`. Running `bash build.sh` without `--dataset` clears any previous preset capacity override in the CMake cache and uses the value in `SLIDE/Config.h` instead; the shipped value is `65536`, suitable for the supplied Wiki10 configuration. This also preserves the option to edit the header and rebuild manually.

**All presets build the same `bin/runme`.** Finish the current run before rebuilding for another dataset; subsequent launches use the most recently built executable. The larger presets increase memory use and initialization cost, so the Wiki10 recommendation of 32 GB RAM should not be treated as a budget for Amazon-Sub or Wiki-325K. If you change the layer sizes, `L`, or `RangePow`, recalculate the required blocks. Check serialization capacity separately when changing model shapes or batch size.

After preparing the data, choose the matching block below. Run it from `SLIDE-ORAM/` with the shared environment activated and an unused run name. The commands perform a build, an input and compiled-capacity check, and then training. The launcher checks the capacity in the actual `bin/runme`; it does not resize or rebuild an insufficient executable automatically.

Amazon-Sub:

```bash
bash build.sh --dataset amzprime
python run_with_set_oram.py Config_amzprime_batch32.csv --threads 1 --run-name amzprime-threads1-batchsize32-repeat1 --check
python run_with_set_oram.py Config_amzprime_batch32.csv --threads 1 --run-name amzprime-threads1-batchsize32-repeat1
```

Wiki-325K:

```bash
bash build.sh --dataset wiki325
python run_with_set_oram.py Config_wiki325_batch32.csv --threads 1 --run-name wiki325-threads1-batchsize32-repeat1 --check
python run_with_set_oram.py Config_wiki325_batch32.csv --threads 1 --run-name wiki325-threads1-batchsize32-repeat1
```

## 3. Run SLIDE-ORAM

**Run all commands in this section from `SLIDE-ORAM/`, with the shared environment activated.** The launcher manages the executable's working directory automatically.

`run_with_set_oram.py` reads a configuration from `SLIDE/`, applies overrides, validates parameters and data headers, and checks the actual compiled ORAM capacity before launching training with the already-built `bin/runme`. Configurations contain `KEY=VALUE` lines. Relative input paths such as `../dataset/Wiki10/train.txt`, including input `weight` paths, resolve from `SLIDE-ORAM/bin/`.

List configurations and check the Wiki10 example without starting training:

```bash
python run_with_set_oram.py --list
python run_with_set_oram.py Config_wiki10_batch32.csv --threads 1 --run-name wiki10-threads1-batchsize32-repeat1 --check
```

`--check` requires prepared data, `bin/runme`, and `stdbuf`. It previews the effective configuration and validates compiled ORAM capacity without creating run files or starting training. It does execute a short build-information query, which exits before model or ORAM initialization.

### Compiled capacity preflight

After rebuilding from the current source, you can inspect the executable's compiled settings directly from `SLIDE-ORAM/`:

```bash
./bin/runme --build-info
```

This command prints JSON containing `format=SLIDE_ORAM_BUILD_INFO_V1`, `schema_version=1`, `ORAM_N`, and `SERIAL_LEN`, then exits without loading a configuration, initializing a model, or running training. It reports values from the executable, rather than the current contents of `SLIDE/Config.h`.

Both `--check` and a normal launch perform this query, with a 10-second timeout, before writing a run directory or generated CSV. Before executing the query, the launcher reads the binary to check for its build-information capability marker. Older binaries without this marker are rejected with a rebuild instruction; the launcher does not pass `--build-info` to them.

The capacity check uses the effective configuration, including any `--set` overrides. For each layer whose training sparsity differs from `1`, it adds the layer's node count and `L * 2^RangePow` hash buckets. It checks that the reported `ORAM_N` is a positive power of two and can hold the total. An insufficient build is rejected with the compiled capacity, required block count, minimum power-of-two capacity, and rebuild guidance for a sufficient preset.

`SERIAL_LEN` is reported but its adequacy is not checked. Preflight also does not validate checkpoint contents, the compiled `LOADWEIGHT` mode, or scientific results. The supplied presets cover configurations with one sparse output layer; passing the block-capacity check is not a guarantee that an arbitrary custom network is correct.

To run the launcher's capacity regression tests from `SLIDE-ORAM/` without training:

```bash
python -B -m unittest discover -s tests -v
```

These tests use temporary header fixtures and mocked subprocesses; they do not run the training executable or validate scientific results.

Start training directly on an x86-64 machine; no TDX or `numactl` prefix is required:

```bash
python run_with_set_oram.py Config_wiki10_batch32.csv --threads 1 --run-name wiki10-threads1-batchsize32-repeat1
```

Keep `--threads 1` to make the recorded OpenMP setting explicit. For custom runtime parameters, repeat `--set KEY=VALUE`, for example `--set Epoch=2`; original configurations are preserved. Compile-time settings cannot be changed with `--set`. This component measures training runtime; its accuracy-evaluation calls are disabled.

The launcher uses `stdbuf -oL` and prefixes each stdout/stderr line with its local receipt time, displaying it and saving it to `console.log`. No additional `tee` or timestamping pipeline is needed. Each run gets a new `dataset/runs/<run-name>/` directory:

| File/directory | Contents |
| --- | --- |
| `console.log` | Timestamped stdout/stderr, status, and overall wall time; input to runtime analysis |
| `training.log` | C++ `logFile` output |
| `savedWeights.npz` | Checkpoint saved at each completed epoch |
| `run.json` | Effective parameters, inputs, hashes, binary build information, capacity calculation, OpenMP settings, status, and exit code |
| `work/` | Executable's working directory |

The launcher saves `SLIDE/Config_<...>__run_<run-name>.csv` and redirects `logFile` and `savedweight` into the run directory. In `run.json`, `binary_build_info` preserves the complete JSON response from the executable, and `oram_capacity` records `required_blocks`, `minimum_oram_n`, `compiled_oram_n`, and the per-layer calculation. `weight` remains an input checkpoint path. Legacy `retrivalfile` entries are ignored. Existing run names are rejected; use explicit names ending in `repeatN`, as in the examples, for analysis. Ctrl+C stops training and records an interrupted run; completed batch measurements can still be analyzed as described below.

## 4. Analyze runtime

Run these commands from **`SLIDE-ORAM/`**, with the shared environment activated. [`../scripts/analyze_slide_oram.py`](../scripts/analyze_slide_oram.py) reads `dataset/runs/<run-name>/console.log` and its matching `run.json`. It produces amortized mean batch time, estimated epoch time, and timing audits without rerunning training or loading datasets. `training.log` is not the timing input, and no figure is generated.

```bash
python ../scripts/analyze_slide_oram.py --dataset wiki10 --list
python ../scripts/analyze_slide_oram.py --dataset wiki10 --batchsize 32 --run wiki10-threads1-batchsize32-repeat1
```

Use `--dataset amazonsub`, `--dataset wiki325`, or `--dataset amazon670k` for the other datasets. Repeat `--run` to average independent runs with matching parameters. `--skip-batches` and `--max-batches` select an analysis window; they do not stop training.

See [Analyzing multiple experiments](../scripts/README.md#analyzing-multiple-experiments) for separate configurations, independent repetitions, output preservation, and comparison-table selection. Its commands run from the artifact root.

**A complete batch needs the next consecutive `START TRAINING Batch` marker as its end boundary.** Initialization and the final batch without that boundary are excluded, even if the process exits successfully. For partial-run analysis, retain enough output to close the batches being measured. A complete timed rehash event must exist in a contributing log, or supply a separately measured event cost with `--rehash-seconds`.

Outputs are written to `../analysis/slide-oram/<dataset>/bs32/threads1/`:

| Output | Contents |
| --- | --- |
| `table_row.csv`, `table_row.md` | Batch time, estimated epoch time, sample counts, and baseline speedup of 1 |
| `batches.csv`, `run_summary.csv` | Batch-level and per-run measurements |
| `result.json` | Parameters, provenance, rehash calibration, and parser audit |

The table reports **one effective thread**. The analyzer's `--threads` option only filters recorded launcher settings; it does not measure parallelism. Batch time includes amortized rehash cost and is displayed in **hours**; structured results also retain seconds for comparison with TENNOR. Epoch time is **extrapolated**, not a measured full epoch.

Rerunning analysis replaces existing outputs. Use `--output-dir analysis/slide-oram/wiki10/<unique-name>` to preserve another selection. Relative analysis path options resolve from the **artifact root**. See the [analysis reference](../scripts/README.md#slide-oram-runtime) for details.

After also analyzing a matching TENNOR run, generate the combined runtime table and speedups:

```bash
python ../scripts/build_runtime_table.py --dataset wiki10 --batchsize 32
```

This reads both components' analysis results and writes Markdown, CSV, and LaTeX tables under `../analysis/runtime/bs32/`. See the [table reference](../scripts/README.md#combined-runtime-table) for explicit result selection when multiple candidates exist.
