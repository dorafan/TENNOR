#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Keep build products and downloaded dependencies in this component.
if [[ -L bin ]]; then
    printf 'Error: bin must be a local build directory, not a symlink.\n' >&2
    exit 1
fi

cmake -S . -B bin -DCMAKE_BUILD_TYPE=Debug "$@"
cmake --build bin --parallel 4 --target runme
