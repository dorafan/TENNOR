# SLIDE-MP: build, run, and analyze accuracy

The paper's accuracy experiments used a **bare-metal x86-64 server with a 64-core AMD EPYC 7763 (Zen 3), without a TEE**. See [Section 7 of the paper](https://arxiv.org/pdf/2605.07160#page=9). **SLIDE-MP supports both AMD and Intel x86-64 CPUs and does not require TDX.** The installation scripts target Ubuntu 24.04.

This component evaluates MP-WTA accuracy using Precision at 1 (P@1). The accuracy analyzer compares these measurements with fresh **50-table WTA logs** collected using the [SLIDE guide](../SLIDE/README.md). The embedded historical WTA reference remains available through explicit `--historical-wta` selection.

## 1. Install dependencies and activate Python

Start at the **artifact root**, the parent of `SLIDE-MP/`:

```bash
bash install_dependencies.sh
source .venv-ae/bin/activate
cd SLIDE-MP
```

The installer installs shared system dependencies, including GCC 13, CMake, and unzip, and creates `.venv-ae/` for downloads and analysis. Internet access and `sudo` (or root) are required.

**Activate the environment before following the commands below.** Activation lasts for the current shell; repeat it in each new terminal. From `SLIDE-MP/`, use `source ../.venv-ae/bin/activate`. If dependencies were installed for another component, reuse the same installation.

## 2. Download data and build

Run these commands from **`SLIDE-MP/`** with the environment activated:

```bash
bash download_datasets.sh
bash build.sh
```

- **`download_datasets.sh`** downloads Wiki10-31K, WikiLSHTC-325K, and Amazon-670K into `dataset/{Wiki10,Wiki325,Amazon}/`. It prepares Amazon-Sub in `dataset/AmzPrime/` from the bundled `../dataset-Amazon-Sub/AmzPrime/{train.zip,test.txt}`. All datasets become `train.txt` / `test.txt` pairs. Existing nonempty pairs are reused; incomplete pairs cause an error. Downloads use the shared `gdown`.
- **`build.sh`** configures a CMake **Debug** build and builds `bin/runme` from the C++ sources in `SLIDE/`, using four build jobs. SLIDE-MP uses C++11, OpenMP, and `-O0`; zlib and CNPY are fetched and built locally under `bin/` in Release mode. First builds need GitHub access. After an interrupted or failed build, rerunning the script removes `bin/` and rebuilds it.

## 3. Run MP-WTA

**Run all commands in this section from `SLIDE-MP/`, with the shared environment activated.** The launcher manages the executable's working directory automatically.

Default configurations are in `SLIDE/`:

| Configuration | Dataset | Batch size |
| --- | --- | ---: |
| `Config_wiki10.csv` | Wiki10-31K | 32 |
| `Config_wiki325.csv` | Wiki-325K | 128 |
| `Config_amz.csv` | Amazon-670K | 128 |
| `Config_amzprime.csv` | Amazon-Sub | 128 |

All defaults use `L=1,1`, `isMPWTA=1`, `isBruteForce=0`, `isHopper=0`, `isLog=1`, `LoadWeight=0`, `epochInit=0`, and `Epoch=30`. The final classification layer uses one hash table.

`run_with_set.py` reads a configuration, validates parameters and dataset headers, applies optional overrides, and launches the already-built `bin/runme`. Configurations contain `KEY=VALUE` lines. Relative input paths such as `../dataset/Wiki10/train.txt` resolve from `SLIDE-MP/bin/`.

Thread selection is left to the OpenMP runtime; the launcher has no `--threads` option and does not set a thread count. It clears inherited OpenMP thread-count, thread-limit, dynamic-team, and binding settings, plus `GOMP_CPU_AFFINITY`, only in the SLIDE-MP child's environment. The parent shell and other components are unchanged. Operating-system CPU limits and externally imposed CPU affinity still apply.

List configurations and check a Wiki10 run without starting training:

```bash
python run_with_set.py --list
python run_with_set.py Config_wiki10.csv --run-name wiki10-repeat1 --check
```

`--check` requires prepared data and `bin/runme`; it previews the effective configuration without writing files or validating scientific results.

Start training:

```bash
python run_with_set.py Config_wiki10.csv --run-name wiki10-repeat1
```

Repeat `--set KEY=VALUE` for custom settings, such as `--set Epoch=1` for a shorter run. Original configurations are preserved. Keep `isHopper=0` so diagnostics stay within the run.

Each run gets a new `dataset/runs/<run-name>/` directory:

| File/directory | Contents |
| --- | --- |
| `training.log` | Epoch headers and P@1 measurements; input to accuracy analysis |
| `console.log` | Captured stdout/stderr for execution diagnostics |
| `savedWeights.npz` | Checkpoint, overwritten at each completed epoch |
| `run.json` | Effective parameters, inputs, hashes, runtime-default thread policy, cleared environment settings, status, and exit code |
| `work/`, `dataset/` | Working directory and local diagnostics, including `dataset/retrival.txt` when produced |

The launcher saves `SLIDE/Config_<...>__run_<run-name>.csv` and redirects output paths into the run directory. Existing run names are rejected; use explicit names ending in `repeatN`, as in the examples, for analysis. Ctrl+C stops training and records an interrupted run.

## 4. Analyze accuracy

Run these commands from **`SLIDE-MP/`**, with the shared environment activated. [`../scripts/analyze_slide_mp.py`](../scripts/analyze_slide_mp.py) reads `dataset/runs/<run-name>/training.log` and its matching `run.json`. It selects completed-epoch P@1 values, averages matching independent runs, and compares them with the fresh WTA log. First collect WTA results using the [SLIDE guide](../SLIDE/README.md). Analysis does not rerun training or load datasets.

```bash
python ../scripts/analyze_slide_mp.py --dataset wiki10 --list
python ../scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1
```

`--list` works without a WTA log. Analysis defaults to `SLIDE/dataset/Wiki10/training_wta.log` and `SLIDE/SLIDE/Config_wiki10_wta.csv` for Wiki10, with corresponding dataset paths for the other presets. Missing WTA logs cause an error. Use `--wta-log PATH` and `--wta-config PATH` to select preserved logs and their actual configurations; all relative paths resolve from the artifact root. Repeat `--wta-log` for independent WTA repetitions, separately from MP-WTA's `--run` selection. One WTA CSV can apply to all logs, or supply one CSV per log in the same order.

Use `--dataset wiki325`, `--dataset amazon670k`, or `--dataset amazonsub` for other supported datasets with available logs. `--run` selects an exact run folder; repeat it to average independent runs with matching parameters. Optional parameter filters, such as `--bucketsize` and `--param KEY=VALUE`, select recorded settings without changing experiments. Runtime-default runs do not record a measured thread count, so select them by run name and collect repetitions on the same machine with the same execution environment.

See [Analyzing multiple experiments](../scripts/README.md#analyzing-multiple-experiments) for separate configurations, independent repetitions, and output preservation. Its commands run from the artifact root.

The paper averages five independent runs (§7.3). To collect that number, repeat training with distinct names ending in `repeat1` through `repeat5`, then select all five with repeated `--run` options. Each contributing run receives equal weight per epoch. The analyzer reports the actual `n_runs` and sample standard deviation when at least two runs contribute. The implementation has no unified seed interface, so individual runs need not be identical.

**Keep each `training.log` together with its original `run.json`.** Only confirmed complete epochs contribute, and the final epoch requires successful-completion metadata. `Epoch 0` is the first trained epoch. Periodic evaluations use two test batches; epoch-end evaluation uses all configured complete test batches. The analyzer selects the last evaluation of each confirmed complete epoch and preserves P@1 on its 0–1 scale.

The WTA runner does not create `run.json`. Its CSV and exact epoch-end iteration count confirm the logged evaluation, including its final epoch; this does not imply that the whole process or checkpoint save succeeded. The modified SLIDE runner now evaluates all complete test batches at every epoch end, matching MP-WTA; rebuild it before collecting new logs. The analyzer defaults to this schedule (`--wta-evaluation full`). Both methods exclude the final incomplete test batch, and `comparison.csv` reports a numerical MP-WTA-minus-WTA difference only when both evaluated sample counts are known and equal. Supply the actual WTA CSV; for older upstream logs, select the legacy schedule documented in the [analysis guide](../scripts/README.md#mp-wta-accuracy).

To use the original embedded historical WTA curve instead of a fresh log, select it explicitly and preserve a separate output directory:

```bash
python ../scripts/analyze_slide_mp.py --dataset wiki10 --run wiki10-repeat1 --historical-wta --output-dir analysis/slide-mp/wiki10/historical-reference
```

Historical values remain labeled as reference data, without an invented repetition count or standard deviation. Do not combine `--historical-wta` with WTA log/configuration options or `--wta-evaluation`.

Outputs are written to `../analysis/slide-mp/<dataset>/`, where the canonical dataset names are `wiki10`, `wiki325`, `amazon670`, and `amzprime`:

| Output | Contents |
| --- | --- |
| `epochs.csv` | MP-WTA per-run, per-epoch P@1 values and source lines |
| `summary.csv` | MP-WTA mean P@1, standard deviation, and contributing run counts |
| `wta_epochs.csv`, `wta_summary.csv` | Measured WTA values, statistics, run counts, and evaluation coverage |
| `comparison.csv` | Both methods' epoch statistics and differences where evaluated sample counts match |
| `wta_reference.csv` | Historical values in `--historical-wta` mode; header only in measured-log mode |
| `analysis.json` | Both methods' selections, parameters, hashes, evaluation audit, and baseline provenance |
| `<dataset>_epoch_accuracy.pdf` / `.png` | Accuracy-versus-epoch figure and preview |

Plotting works without a display server. Rerunning analysis replaces existing outputs; use `--output-dir analysis/slide-mp/wiki10/<unique-name>` to preserve another selection. Relative analysis path options resolve from the **artifact root**. See the [analysis reference](../scripts/README.md#mp-wta-accuracy) for details.
