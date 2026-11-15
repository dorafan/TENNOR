#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ -r /etc/os-release ]] || fail 'This script requires Ubuntu 24.04 x86-64.'
. /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 ]] ||
    fail 'This script targets Ubuntu 24.04; use that release for the documented setup.'
[[ "$(uname -m)" == x86_64 ]] || fail 'The C++ components require an x86-64 machine.'

sudo_prefix=()
if [[ $EUID -ne 0 ]]; then
    command -v sudo >/dev/null || fail 'Install sudo or run this installer as root.'
    sudo_prefix=(sudo)
fi

apt_packages=(
    build-essential gcc-13 g++-13 cmake git ca-certificates
    python3 python3-venv unzip numactl
)
"${sudo_prefix[@]}" apt-get update
"${sudo_prefix[@]}" apt-get install -y "${apt_packages[@]}"

# Use system Python even if the caller has activated a different environment.
/usr/bin/python3 -m venv .venv-ae
.venv-ae/bin/python -m pip install \
    'gdown==5.2.2' 'numpy==1.26.4' 'pandas==2.2.3' 'matplotlib==3.9.2'
.venv-ae/bin/python -m pip check

# Record the resolved packages for the user's environment report.
.venv-ae/bin/python -m pip freeze > .venv-ae/requirements-resolved.txt
dpkg-query -W "${apt_packages[@]}" > .venv-ae/system-packages.txt

printf '\nDependencies installed for TENNOR, SLIDE-ORAM, SLIDE-MP, and SLIDE WTA.\n'
printf 'Download scripts automatically use .venv-ae; activation is not required.\n'
printf 'Next, for example: bash TENNOR/download_datasets.sh\n'
printf 'Then build: bash TENNOR/build.sh\n'
printf 'For plotting, use .venv-ae/bin/python with a script in scripts/.\n'
