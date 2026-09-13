# Third-Party Notices

The code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).

Portions of this repository are derived from, or build against, third-party
open-source projects. Those portions remain subject to their original licenses,
which are listed below. This file is provided to satisfy the attribution
requirements of those licenses.

---

## SLIDE (HashingDeepLearning)

- **Upstream:** <https://github.com/keroro824/HashingDeepLearning>
- **License:** MIT License
- **Reference commit:** `2d10d46b5f6f1eda5d19f27038a596446fc17cee` (pre-HugePages)

The `TENNOR/`, `SLIDE-ORAM/`, and `SLIDE-MP/` trees are built on top of the
SLIDE codebase. Files derived from SLIDE include (per directory, under
`SLIDE/`): `Bucket.*`, `Config.h`, `DensifiedMinhash.*`, `DensifiedWtaHash.*`,
`LSH.*`, `Layer.*`, `MurmurHash.*`, `Network.*`, `Node.*`, `WtaHash.*`,
`srp.*`, `InputSpecificActiveNetworkFactory.h`, `main.cpp`, `Makefile`, the
`Config_*.csv` configuration files, and the per-directory `CMakeLists.txt`.

These files have been modified for this work. Files such as
`Layer_New.*`, `Network_New.*`, `LSH_New.h`, `Oht.*`, and `Requester.*` are
substantially new code written for TENNOR, but inherit structure and
interfaces from SLIDE.

`MurmurHash.*` originates from MurmurHash by Austin Appleby, which its author
placed in the public domain; it is redistributed here as it appears in SLIDE.

## Snoopy

- **Upstream:** <https://github.com/ucbrise/snoopy>
- **License:** Apache License, Version 2.0

The oblivious primitive implementations used by TENNOR are taken from Snoopy
and integrated into this repository as:

- `TENNOR/SLIDE/obl_primitives.h`, `TENNOR/SLIDE/obl.h`
- `SLIDE-ORAM/SLIDE/obl_primitives.h`, `SLIDE-ORAM/SLIDE/obl.h`

(The `snoopy::` namespace in these files marks the upstream-derived code.)
A copy of the Apache License, Version 2.0 is available at
<https://www.apache.org/licenses/LICENSE-2.0>.

---

## Build-time dependencies (fetched, not vendored)

The CMake build fetches the following projects at configure time via
`ExternalProject_Add`. Their source is **not** included in this repository;
each is governed by its own license.

| Project | Upstream | License |
| --- | --- | --- |
| cnpy | <https://github.com/sarthakpati/cnpy> | MIT License |
| zlib (v1.2.11) | <https://github.com/madler/zlib> | zlib License |

OpenMP support is provided by the compiler toolchain (GCC) and is not
redistributed here.

---

## Datasets

The public extreme multi-label classification datasets used in the paper
(Wiki10-31K, Amazon-670K, WikiLSHTC-325K) are **not** redistributed in this
repository. They must be obtained from the Extreme Classification Repository:

- <http://manikvarma.org/downloads/XC/XMLRepository.html>

`dataset-Amazon-Sub/` contains the raw data and the generation script for the
derived **Amazon-Sub** subset used in the paper. Its contents are subject to
the terms of the upstream source dataset from which it was derived.
