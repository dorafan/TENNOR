#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

# BoW links from http://manikvarma.org/downloads/XC/XMLRepository.html
# Use gdown 5.2.2 with direct download URLs to preserve WikiLSHTC's resourcekey.
# Prefer the shared environment installed at the AE root; no activation needed.
gdown_bin='../.venv-ae/bin/gdown'
if [[ ! -x "$gdown_bin" ]]; then
    gdown_bin=$(command -v gdown) || fail 'Run bash ../install_dependencies.sh first.'
fi
command -v unzip >/dev/null || fail 'Install unzip first: sudo apt install unzip'
[[ "$("$gdown_bin" --version)" == "gdown 5.2.2 at "* ]] ||
    fail 'Use gdown==5.2.2; rerun bash ../install_dependencies.sh.'

prepare_dataset() {
    local folder="$1" archive_name="$2" prefix="$3" url="$4"
    local directory="dataset/$folder"
    local archive="$directory/$archive_name"
    local member name members
    local -a trains=() tests=()

    if [[ -f "$directory/train.txt" && -s "$directory/train.txt" &&
          -f "$directory/test.txt" && -s "$directory/test.txt" ]]; then
        printf 'Already prepared: %s\n' "$directory"
        return
    fi
    if [[ -e "$directory/train.txt" || -L "$directory/train.txt" ||
          -e "$directory/test.txt" || -L "$directory/test.txt" ]]; then
        fail "$directory: incomplete train.txt/test.txt pair; move existing files aside first."
    fi

    mkdir -p "$directory"
    if [[ ! -s "$archive" ]]; then
        "$gdown_bin" --continue --no-cookies "$url" -O "$archive"
        [[ -s "$archive" ]] || fail "Download did not produce a nonempty ZIP: $archive"
    fi

    # Locate the two data files without depending on the ZIP's top-level folder.
    members=$(unzip -Z1 "$archive")
    while IFS= read -r member; do
        case "/$member" in
            */__MACOSX/*|*/._*) continue ;;
        esac
        name=${member##*/}
        case "${name,,}" in
            train.txt|"${prefix}_train.txt") trains+=("$member") ;;
            test.txt|"${prefix}_test.txt") tests+=("$member") ;;
        esac
    done <<< "$members"

    [[ ${#trains[@]} -eq 1 && ${#tests[@]} -eq 1 ]] ||
        fail "$archive: expected exactly one train file and one test file; inspect the ZIP names."
    [[ "$(dirname -- "${trains[0]}")" == "$(dirname -- "${tests[0]}")" ]] ||
        fail "$archive: train and test are in different folders."

    # Publish the final names only after both extractions succeed.
    printf 'Preparing: %s\n' "$directory"
    unzip -p "$archive" "${trains[0]}" > "$directory/train.txt.part"
    unzip -p "$archive" "${tests[0]}" > "$directory/test.txt.part"
    [[ -s "$directory/train.txt.part" && -s "$directory/test.txt.part" ]] ||
        fail "$archive: extracted train or test file is empty."
    mv -- "$directory/train.txt.part" "$directory/train.txt"
    mv -- "$directory/test.txt.part" "$directory/test.txt"
    printf 'Ready: %s/train.txt and %s/test.txt\n' "$directory" "$directory"
}

# Wiki10-31K
prepare_dataset Wiki10 Wiki10.bow.zip wiki10 \
    'https://drive.google.com/uc?export=download&id=1-sE0GoT-eE18S9e-B8nQr8j7LdYexGHX'

# WikiLSHTC-325K (keep the resourcekey).
prepare_dataset Wiki325 WikiLSHTC.zip wikilshtc \
    'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGSHE1SWx4TVRva3c&resourcekey=0-ZGNqdLuqttRdnAj-U0bktA'

# Amazon-670K
prepare_dataset Amazon Amazon670K.bow.zip amazon \
    'https://drive.google.com/uc?export=download&id=1TLaXCNB_IDtLhk4ycOnyud0PswWAW6hR'

# Amazon-Sub: use the author's bundled subset without regenerating it.
source_directory='../dataset-Amazon-Sub/AmzPrime'
directory='dataset/AmzPrime'

if [[ -f "$directory/train.txt" && -s "$directory/train.txt" &&
      -f "$directory/test.txt" && -s "$directory/test.txt" ]]; then
    printf 'Already prepared: %s\n' "$directory"
elif [[ -e "$directory/train.txt" || -L "$directory/train.txt" ||
        -e "$directory/test.txt" || -L "$directory/test.txt" ]]; then
    fail "$directory: incomplete train.txt/test.txt pair; move existing files aside first."
else
    [[ -f "$source_directory/train.zip" && -s "$source_directory/train.zip" &&
       -f "$source_directory/test.txt" && -s "$source_directory/test.txt" ]] ||
        fail "Amazon-Sub requires $source_directory/train.zip and $source_directory/test.txt."
    mkdir -p "$directory"
    printf 'Preparing Amazon-Sub: %s\n' "$directory"
    unzip -p "$source_directory/train.zip" train.txt > "$directory/train.txt.part"
    cp -- "$source_directory/test.txt" "$directory/test.txt.part"
    [[ -s "$directory/train.txt.part" && -s "$directory/test.txt.part" ]] ||
        fail 'Amazon-Sub: prepared train or test file is empty.'
    mv -- "$directory/train.txt.part" "$directory/train.txt"
    mv -- "$directory/test.txt.part" "$directory/test.txt"
    printf 'Ready: %s/train.txt and %s/test.txt\n' "$directory" "$directory"
fi
