#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

# Reuse SLIDE-MP's prepared inputs without sharing output directories.
# Return 1 only when neither this component nor SLIDE-MP has a complete pair.
reuse_dataset() {
    local folder="$1" directory="dataset/$1"
    local shared_directory="../SLIDE-MP/dataset/$1"
    if [[ -L dataset || -L "$directory" ]]; then
        fail "$directory: use a real directory so WTA outputs stay separate from SLIDE-MP."
    fi
    if [[ -f "$directory/train.txt" && -s "$directory/train.txt" &&
          -f "$directory/test.txt" && -s "$directory/test.txt" ]]; then
        printf 'Already prepared: %s\n' "$directory"
        return 0
    fi
    if [[ -e "$directory/train.txt" || -L "$directory/train.txt" ||
          -e "$directory/test.txt" || -L "$directory/test.txt" ]]; then
        fail "$directory: incomplete train.txt/test.txt pair; move existing files aside first."
    fi
    if [[ -f "$shared_directory/train.txt" && -s "$shared_directory/train.txt" &&
          -f "$shared_directory/test.txt" && -s "$shared_directory/test.txt" ]]; then
        mkdir -p "$directory" || fail "Cannot create $directory"
        ln -s "../../../SLIDE-MP/dataset/$folder/train.txt" "$directory/train.txt" ||
            fail "Cannot link $directory/train.txt"
        ln -s "../../../SLIDE-MP/dataset/$folder/test.txt" "$directory/test.txt" ||
            fail "Cannot link $directory/test.txt"
        printf 'Reusing SLIDE-MP inputs: %s\n' "$directory"
        return 0
    fi
    return 1
}

download_archive() {
    local url="$1" archive="$2"
    # Prefer the shared environment; no activation is needed. Check gdown only
    # when a download is needed, so prepared inputs also work offline.
    local gdown_bin='../.venv-ae/bin/gdown'
    if [[ ! -x "$gdown_bin" ]]; then
        gdown_bin=$(command -v gdown) || fail 'Run bash ../install_dependencies.sh first.'
    fi
    [[ "$("$gdown_bin" --version)" == "gdown 5.2.2 at "* ]] ||
        fail 'Use gdown==5.2.2; rerun bash ../install_dependencies.sh.'
    "$gdown_bin" --continue --no-cookies "$url" -O "$archive"
    [[ -s "$archive" ]] || fail "Download did not produce a nonempty ZIP: $archive"
}

require_unzip() {
    command -v unzip >/dev/null || fail 'Install unzip first: sudo apt install unzip'
}

# Keep the archive URLs and extraction rules aligned with SLIDE-MP's downloader.
# BoW links: http://manikvarma.org/downloads/XC/XMLRepository.html
prepare_dataset() {
    local folder="$1" archive_name="$2" prefix="$3" url="$4"
    local directory="dataset/$folder"
    local archive="$directory/$archive_name"
    local member name members
    local -a trains=() tests=()

    if reuse_dataset "$folder"; then
        return
    fi
    require_unzip
    mkdir -p "$directory"
    if [[ ! -s "$archive" ]]; then
        download_archive "$url" "$archive"
    fi

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

    printf 'Preparing: %s\n' "$directory"
    unzip -p "$archive" "${trains[0]}" > "$directory/train.txt.part"
    unzip -p "$archive" "${tests[0]}" > "$directory/test.txt.part"
    [[ -s "$directory/train.txt.part" && -s "$directory/test.txt.part" ]] ||
        fail "$archive: extracted train or test file is empty."
    mv -- "$directory/train.txt.part" "$directory/train.txt"
    mv -- "$directory/test.txt.part" "$directory/test.txt"
    printf 'Ready: %s/train.txt and %s/test.txt\n' "$directory" "$directory"
}

prepare_dataset Wiki10 Wiki10.bow.zip wiki10 \
    'https://drive.google.com/uc?export=download&id=1-sE0GoT-eE18S9e-B8nQr8j7LdYexGHX'

prepare_dataset Wiki325 WikiLSHTC.zip wikilshtc \
    'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGSHE1SWx4TVRva3c&resourcekey=0-ZGNqdLuqttRdnAj-U0bktA'

prepare_dataset Amazon Amazon670K.bow.zip amazon \
    'https://drive.google.com/uc?export=download&id=1TLaXCNB_IDtLhk4ycOnyud0PswWAW6hR'

# Amazon-Sub: use the author's bundled subset without regenerating it.
if ! reuse_dataset AmzPrime; then
    source_directory='../dataset-Amazon-Sub/AmzPrime'
    directory='dataset/AmzPrime'
    [[ -f "$source_directory/train.zip" && -s "$source_directory/train.zip" &&
       -f "$source_directory/test.txt" && -s "$source_directory/test.txt" ]] ||
        fail "Amazon-Sub requires $source_directory/train.zip and $source_directory/test.txt."
    require_unzip
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
