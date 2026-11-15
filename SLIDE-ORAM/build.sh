#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
Usage: bash build.sh [--dataset wiki10|amzprime|wiki325]

Compile-time ORAM capacity presets for the supplied batch32 configurations:
  wiki10     65536 blocks
  amzprime   524288 blocks (Amazon-Sub)
  wiki325    2097152 blocks (Wiki-325K)

Without --dataset, use ORAM_N from SLIDE/Config.h and clear any previous
preset override. Each build updates the same bin/runme executable.
EOF
}

dataset=""
if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
    usage
    exit 0
elif [ "$#" -eq 2 ] && [ "$1" = "--dataset" ]; then
    dataset="$2"
    if [ -z "$dataset" ]; then
        usage >&2
        exit 2
    fi
elif [ "$#" -ne 0 ]; then
    usage >&2
    exit 2
fi

case "$dataset" in
    "")       oram_n="" ;;
    wiki10)   oram_n=65536 ;;
    amzprime) oram_n=524288 ;;
    wiki325)  oram_n=2097152 ;;
    *) printf 'Unknown dataset preset: %s\n' "$dataset" >&2; usage >&2; exit 2 ;;
esac

if [ -n "$oram_n" ]; then
    printf 'Building SLIDE-ORAM for %s with ORAM_N=%s.\n' "$dataset" "$oram_n"
else
    printf 'Building SLIDE-ORAM with ORAM_N from SLIDE/Config.h (no preset override).\n'
fi

if [ -f .build-incomplete ]; then
    printf 'Previous build did not finish successfully; removing bin/ and rebuilding.\n'
    rm -rf -- bin
fi

touch .build-incomplete
cmake -S . -B bin -DCMAKE_BUILD_TYPE=Debug "-DSLIDE_ORAM_N=${oram_n}"
cmake --build bin --parallel 4 --target runme
rm -f -- .build-incomplete
