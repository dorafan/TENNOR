#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [ -f .build-incomplete ]; then
    printf 'Previous build did not finish successfully; removing bin/ and rebuilding.\n'
    rm -rf -- bin
fi

touch .build-incomplete
cmake -S . -B bin -DCMAKE_BUILD_TYPE=Debug
cmake --build bin --parallel 4 --target runme
rm -f -- .build-incomplete
