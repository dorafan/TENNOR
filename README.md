# TENNOR Artifact: Build & Run Guide

This repository contains the research artifact for **TENNOR**.

The artifact includes:

1. the full implementation of TENNOR;
2. the integrated baselines used in the paper;
3. the implementation of MP-WTA baseline for accuracy test;
4. the configuration files used for the reported experiments;
5. dataset preparation files for derived datasets; and
6. Python plotting/table-generation scripts based on pre-collected experiment outputs.

Due to the hardware requirements of Intel TDX and the large memory footprint of some experiments, full end-to-end reproduction may not be feasible on all platforms. This README documents the required environment, repository layout, build instructions, representative run commands, and the workflow for regenerating figures and tables from the included pre-collected outputs.

---

## 1. Hardware and Software Environment

### 1.1 Performance Experiments

The full TENNOR pipeline and the TEE-based performance experiments require an Intel TDX-enabled platform with sufficient memory.

The experiments reported in the paper were conducted on a **Google Cloud Confidential C3** machine with the following configuration:

- **CPU:** 2 x 22-core / 44-thread processors, Intel Sapphire Rapids
- **Memory:** 352 GB
- **TEE:** Intel TDX
- **OS:** Ubuntu 24.04 LTS confidential image
- **ISA support:** AVX, AVX2, AVX-512

Implementation and toolchain:

- **Language:** C++
- **Compiler:** GCC 13.3.0 on Ubuntu 24.04
- **Build system:** CMake
- **Parallelism:** OpenMP

TENNOR uses x86-specific oblivious primitive implementations and should be built and evaluated on an x86 platform.

### 1.2 Accuracy Experiments

The accuracy component, implemented in `SLIDE-MP`, can run on commodity x86 hardware. However, large configurations, such as standard WTA with many hash tables, may require substantial memory.

---

## 2. Datasets

The paper uses public extreme multi-label classification datasets from the Extreme Classification Repository:

- XML Repository: <http://manikvarma.org/downloads/XC/XMLRepository.html>

The public datasets used in the paper include:

- **Wiki10-31K**
- **Amazon-670K**
- **WikiLSHTC-325K**

Please download these datasets directly from the XML Repository and place them under the dataset directory expected by the corresponding configuration file.

For derived datasets, such as **Amazon-Sub**, this artifact includes the exact raw data used in the paper and the scripts required to generate the derived subset.

---

## 3. Repository Structure

The repository is organized as follows:

```text
.
|-- TENNOR/                 # TENNOR implementation
|-- SLIDE-ORAM/             # Baseline: SLIDE with ORAM-backed access
|-- SLIDE-MP/               # MP-WTA implementation for algorithmic accuracy experiments
|-- scripts/                # Plot generation scripts using pre-collected outputs
|-- dataset-Amazon-Sub/     # Raw data and construction scripts for Amazon-Sub
`-- README.md
```

Please adjust dataset paths in the configuration files before running experiments.

---

## 4. Dependencies

The artifact was tested on Ubuntu 24.04 with GCC 13.3.0.

Install the common build dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake gcc-13 g++-13 python3 python3-venv python3-pip numactl zlib1g-dev
```

The implementation also depends on:

- CNPY
- Zlib
- OpenMP

GCC normally provides OpenMP support through `-fopenmp`. Depending on the platform, additional OpenMP runtime packages may be required.

---

## 5. Build Instructions

The artifact uses CMake. Each system can be built from its corresponding directory.

### 5.1 Build TENNOR

```bash
cd TENNOR
mkdir -p bin
cd bin
cmake ..
make
```

### 5.2 Build SLIDE-ORAM

```bash
cd SLIDE-ORAM
mkdir -p bin
cd bin
cmake ..
make
```

### 5.3 Build SLIDE-MP

```bash
cd SLIDE-MP
mkdir -p bin
cd bin
cmake ..
make
```

`SLIDE-MP` is used for algorithmic accuracy experiments, including MP-WTA and standard WTA configurations.

---

## 6. Running Experiments

Before running any experiment, make sure that the dataset path in the corresponding configuration file is correct.

A typical run command has the following form:

```bash
./runme ../SLIDE/Config_datasetname_settings.csv
```

Replace `Config_datasetname_settings.csv` with the configuration file corresponding to the desired dataset and setting.

### 6.1 Running TENNOR

Example:

```bash
cd TENNOR/bin

numactl --cpunodebind=0 --interleave=all \
env OMP_NUM_THREADS=44 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=threads \
./runme ../SLIDE/Config_wiki325_44_batch32.csv 2>&1 | tee -a ../dataset/output_wiki325_44_32.txt
```

The exact configuration files used in the paper are included in the repository.

### 6.2 Running SLIDE-ORAM

Example:

```bash
cd SLIDE-ORAM/bin

numactl --cpunodebind=0 --interleave=all \
env OMP_NUM_THREADS=44 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=threads \
./runme ../SLIDE/Config_datasetname_settings.csv 2>&1 | tee -a ../dataset/output_baseline.txt
```

### 6.3 Running SLIDE-MP

Example:

```bash
cd SLIDE-MP/bin

./runme ../SLIDE/Config_datasetname_settings.csv
```

`SLIDE-MP` can be used to evaluate MP-WTA accuracy as well as standard WTA configurations.

---

## 7. Figure Generation

The `scripts/` directory contains Python scripts for regenerating the figures in the paper.

These scripts are **not** end-to-end experiment automation scripts. They do not launch training runs, collect runtime logs, or automatically parse raw system outputs from scratch. Instead, they generate plots from pre-collected experiment outputs included in the artifact.

To regenerate a figure, run the corresponding Python script directly:

```bash
python3 scripts/<script_name>.py
```

To reproduce the underlying experiment outputs from scratch, one should first run the corresponding experiments on a compatible platform, collect the raw outputs, and then update or use the plotting/table-generation scripts accordingly.

---

## 8. Reproducibility Notes

### Hardware Requirements

Full end-to-end reproduction of the TEE-based experiments requires:

- an Intel TDX-enabled platform;
- sufficient memory for large padded data structures;
- x86 support for the oblivious primitive implementation; and
- an environment compatible with OpenMP parallel execution.

The accuracy experiments in `SLIDE-MP` are less restrictive and can run on commodity x86 hardware, although large hash-table configurations may still require substantial memory.

### Determinism

Runtime measurements may vary across runs due to:

- multi-thread scheduling;
- CPU frequency scaling;
- NUMA placement;
- cloud VM variability; and
- TDX-related overheads.

When using CPU affinity or NUMA pinning, we recommend documenting the exact options used for each run.

### Configuration Files

The repository includes the configuration files used for the reported experiments. Before running an experiment, check that:

1. the dataset path is correct;
2. the batch size matches the intended setting;
3. the number of threads matches the hardware configuration; and
4. the LSH/WTA/MP-WTA parameters match the desired experiment.

---

## 9. Data Provenance and External Code

### Public Datasets

Public datasets are obtained from the Extreme Classification Repository:

- <http://manikvarma.org/downloads/XC/XMLRepository.html>

### Amazon-Sub

For derived datasets such as **Amazon-Sub**, this artifact includes the exact raw data used in the paper and the scripts used to generate the derived subset.

### SLIDE

The baseline implementations build on the open-source SLIDE codebase:

- <https://github.com/keroro824/HashingDeepLearning>

If needed, the original SLIDE codebase can be reverted to commit:

```text
2d10d46b5f6f1eda5d19f27038a596446fc17cee
```

This avoids the HugePages optimization, which may not be available on all platforms.

### Snoopy

TENNOR uses oblivious primitive implementations from Snoopy and integrates the required components into this repository:

- <https://github.com/ucbrise/snoopy>

---

## 10. Artifact Scope

This repository provides:

- source code for TENNOR;
- integrated baselines;
- source code for MP-WTA version
- experiment configuration files;
- dataset preparation files for derived datasets;
- representative build and run instructions; and
- plot-generation scripts based on pre-collected experiment outputs.

Because some experiments require Intel TDX and substantial memory, not every user platform will be able to reproduce the full performance evaluation end to end. The included code, configuration files, pre-collected outputs, and plotting-generation scripts are provided to make the evaluation process as transparent and reproducible as possible.


---

## 11. License

The code in this repository is released under the MIT License; see [`LICENSE`](LICENSE).

Parts of this repository are derived from third-party open-source projects
(SLIDE, Snoopy, cnpy, zlib), which remain under their own licenses.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the full list of
components, upstream sources, and applicable licenses.
