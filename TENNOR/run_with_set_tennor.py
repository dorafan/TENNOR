#!/usr/bin/env python3
"""Run a TENNOR configuration with repeatable --set KEY=VALUE overrides."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
SLIDE = ROOT / "SLIDE"
DATASET = ROOT / "dataset"
BIN = ROOT / "bin"
KEYS = set("RangePow K L Sparsity Batchsize Rehash Rebuild InputDim totRecords "
           "totRecordsTest Epoch Lr Stepsize numLayer logFile sizesOfLayers "
           "trainData testData weight savedweight LoadWeight epochInit BUCKETSIZE "
           "binsize isBruteForce NNSorMPS BFnumber isMPWTA isRandom isHopper "
           "isLog threatNum".split())
# Some existing CSVs contain this field, but TENNOR's C++ parser does not read it.
LEGACY_KEYS = {"retrivalfile"}


def fail(message):
    raise ValueError(message)


def contained(path, parent):
    return path == parent or parent in path.parents


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_config(path):
    values = {}
    text = path.read_text(encoding="utf-8-sig")
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            fail(f"{path.name}:{number}: use whole-line comments, not inline '#'.")
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key not in (KEYS | LEGACY_KEYS) or not value:
            fail(f"{path.name}:{number}: expected a known, nonempty key=value.")
        if value.endswith(","):
            fail(f"{path.name}:{number}: trailing CSV comma; this format is key=value.")
        if key in values and values[key] != value:
            fail(f"{path.name}:{number}: conflicting duplicate key {key}.")
        values[key] = value
    return values, text


def read_overrides(settings):
    overrides = {}
    for setting in settings:
        key, sep, value = setting.partition("=")
        key, value = key.strip(), value.strip()
        if key == "retrivalfile":
            fail("TENNOR does not read retrivalfile. With isHopper=0 its retrieval log "
                 "is always dataset/retrival.txt inside this run directory.")
        if not sep or key not in KEYS or not value:
            fail(f"Invalid --set {setting!r}: expected a known, nonempty KEY=VALUE.")
        if any(character in setting for character in ("#", "\n", "\r", "\0")):
            fail("--set cannot contain '#', newlines, or NUL characters.")
        if value.endswith(","):
            fail(f"--set {key}: remove the trailing comma.")
        overrides[key] = value  # Repeated keys use the last command-line value.
    return overrides


def integer(values, key, default=None, minimum=0):
    raw = values.get(key, default)
    if raw is None or not re.fullmatch(r"[0-9]+", str(raw)):
        fail(f"{key} must be an integer >= {minimum}.")
    value = int(raw)
    if value < minimum:
        fail(f"{key} must be >= {minimum}.")
    return value


def source_path(raw):
    path = Path(raw)
    return (path if path.is_absolute() else BIN / path).resolve()


def check_data(path, filename, required_rows, features, labels):
    if not contained(path, DATASET.resolve()) or path.name != filename:
        fail(f"Expected this component's dataset/<name>/{filename}: {path}")
    if not path.is_file():
        fail(f"Missing data file: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        header = stream.readline().split()
    if len(header) != 3 or not all(re.fullmatch(r"[0-9]+", x) for x in header):
        fail(f"Expected the header 'rows features labels' in {path}.")
    rows, dimension, label_count = map(int, header)
    if rows < required_rows:
        fail(f"{path}: header has {rows} rows; this run needs at least {required_rows}.")
    if dimension != features or label_count != labels:
        fail(f"{path}: header dimensions ({dimension}, {label_count}) disagree with "
             f"InputDim / output layer ({features}, {labels}).")
    return {"path": str(path), "header_rows": rows,
            "features": dimension, "labels": label_count}


def validate(values):
    for key in ("numLayer", "RangePow", "K", "L", "Sparsity", "sizesOfLayers",
                "trainData", "testData", "Lr"):
        if key not in values:
            fail(f"Missing required config key: {key}")
    layers = integer(values, "numLayer", minimum=1)
    arrays = {}
    for key in ("RangePow", "K", "L", "sizesOfLayers"):
        parts = [x.strip() for x in values[key].split(",")]
        if len(parts) != layers or not all(re.fullmatch(r"[0-9]+", x) for x in parts):
            fail(f"{key} must contain numLayer={layers} integers.")
        arrays[key] = list(map(int, parts))
        if any(x < 1 for x in arrays[key]):
            fail(f"All {key} entries must be positive.")
    if any(x > 30 for x in arrays["RangePow"]):
        fail("RangePow must be <= 30 for the current signed-int bucket shift.")
    try:
        sparsity = [float(x.strip()) for x in values["Sparsity"].split(",")]
        learning_rate = float(values["Lr"])
    except ValueError:
        fail("Sparsity and Lr must be numeric.")
    if len(sparsity) != 2 * layers or not all(0 < x <= 1 for x in sparsity):
        fail("Sparsity must contain 2*numLayer values in (0, 1].")
    if not 0 < learning_rate < float("inf"):
        fail("Lr must be finite and positive.")
    batch = integer(values, "Batchsize", minimum=1)
    train_count = integer(values, "totRecords", minimum=batch)
    test_count = integer(values, "totRecordsTest", minimum=batch)
    features = integer(values, "InputDim", minimum=1)
    epochs = integer(values, "Epoch", minimum=1)
    first_epoch = integer(values, "epochInit", default=0)
    integer(values, "Stepsize", minimum=1)
    for key in ("Rehash", "Rebuild"):
        integer(values, key, minimum=batch)
    for key, default in (("BUCKETSIZE", 256), ("binsize", 8)):
        integer(values, key, default=default, minimum=1)
    for key, default in (("LoadWeight", 0), ("isBruteForce", 0), ("isMPWTA", 0),
                         ("isRandom", 0), ("isHopper", 0), ("isLog", 1), ("NNSorMPS", 0)):
        if integer(values, key, default=default) not in (0, 1):
            fail(f"{key} must be 0 or 1.")
    integer(values, "BFnumber", default=0)
    integer(values, "threatNum", default=1)  # 0 means the OpenMP runtime maximum.
    if first_epoch >= epochs:
        fail("epochInit must be smaller than Epoch; otherwise no training runs.")
    train = source_path(values["trainData"])
    test = source_path(values["testData"])
    if train.parent != test.parent:
        fail("train.txt and test.txt must be in the same dataset directory.")
    inputs = [check_data(train, "train.txt", train_count // batch * batch,
                         features, arrays["sizesOfLayers"][-1]),
              check_data(test, "test.txt", test_count // batch * batch,
                         features, arrays["sizesOfLayers"][-1])]
    weights = None
    if integer(values, "LoadWeight", default=0):
        if "weight" not in values:
            fail("LoadWeight=1 requires weight=<checkpoint path>.")
        weights = source_path(values["weight"])
        if not contained(weights, DATASET.resolve()) or not weights.is_file():
            fail(f"Checkpoint must exist within this component's dataset: {weights}")
    return inputs, weights


def prepare_effective(values, overrides, inputs, weights, run_dir):
    effective = dict(values)
    legacy_retrieval = effective.pop("retrivalfile", None)
    effective.setdefault("threatNum", "1")  # Matches THREATNUM in main.cpp.
    effective.update(trainData=inputs[0]["path"], testData=inputs[1]["path"])
    effective["isHopper"] = overrides.get("isHopper", "0")
    outputs = {}
    defaults = {"logFile": "training.log", "savedweight": "savedWeights.npz"}
    for key, default in defaults.items():
        path = Path(overrides.get(key, default))
        path = (path if path.is_absolute() else run_dir / path).resolve()
        if not contained(path, run_dir) or path == run_dir:
            fail(f"--set {key}: output must be a file inside this run directory: {run_dir}")
        if contained(path, run_dir / "work"):
            fail(f"--set {key}: work/ is reserved for the program's working directory.")
        if contained(path, run_dir / "dataset"):
            fail(f"--set {key}: dataset/ is reserved for C++ diagnostic logs; use another subdirectory.")
        for reserved in (run_dir / "console.log", run_dir / "run.json"):
            if contained(path, reserved) or contained(reserved, path):
                fail(f"--set {key}: output conflicts with the launcher's {reserved.name}.")
        if any(contained(path, other) or contained(other, path) for other in outputs.values()):
            fail(f"--set {key}: output paths must be distinct files, not parents of each other.")
        if any(character in str(path) for character in ("#", "\n", "\r")):
            fail(f"--set {key}: the C++ config parser cannot handle '#' or newlines in paths.")
        outputs[key] = path
        effective[key] = str(path)
    if weights is not None:
        effective["weight"] = str(weights)
    elif "weight" in overrides:
        effective["weight"] = str(source_path(overrides["weight"]))
    else:
        effective["weight"] = effective["savedweight"]
    notices = []
    if legacy_retrieval is not None:
        notices.append("Ignored legacy retrivalfile from the original CSV: TENNOR does not read "
                       "this field, so it is omitted from the generated config.")
    if int(effective["isHopper"]) == 1:
        notices.append("isHopper=1: C++ writes LSH/retrieval diagnostics to "
                       "/scratch/zqu4/SLIDE-MP/dataset; those files are outside this run directory.")
    elif int(values.get("isHopper", "0")) != 0:
        notices.append("Using isHopper=0 so diagnostic logs stay inside this run directory.")
    return effective, outputs, notices


def resolve_threads(requested, threat_num):
    if requested is not None:
        threads, source = requested, "--threads"
    elif "OMP_NUM_THREADS" in os.environ:
        try:
            threads = int(os.environ["OMP_NUM_THREADS"])
        except ValueError:
            fail("OMP_NUM_THREADS must be a positive integer, or supply --threads.")
        source = "OMP_NUM_THREADS"
    else:
        threads = threat_num or None
        source = "threatNum" if threat_num else "OpenMP runtime default"
    if threads is not None and threads < 1:
        fail("--threads / OMP_NUM_THREADS must be positive.")
    return threads, source


def stop_child(child):
    if child.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="Config_*.csv name, or path inside SLIDE")
    parser.add_argument("--list", action="store_true", help="list original config filenames")
    parser.add_argument("--check", action="store_true", help="check config/headers/binary; do not write or run")
    parser.add_argument("--threads", type=int,
                        help="OpenMP threads (default: OMP_NUM_THREADS, then threatNum; 0 threatNum uses runtime default)")
    parser.add_argument("--run-name", help="new output directory name under dataset/runs")
    parser.add_argument("--set", dest="settings", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config variable; repeat for multiple variables (last value wins)")
    args = parser.parse_args()
    if args.list:
        if args.config or args.check or args.run_name or args.threads is not None or args.settings:
            parser.error("--list must be used on its own")
        for path in sorted(SLIDE.glob("Config_*.csv")):
            if "__run_" not in path.stem:
                print(path.name)
        return 0
    if not args.config:
        parser.error("provide a config, or use --list")
    if sys.platform != "linux":
        fail("This launcher targets Linux, matching the documented build environment.")
    config = Path(args.config)
    if not config.is_absolute():
        config = (ROOT / config) if config.parts[0] == "SLIDE" else (SLIDE / config)
    config = config.resolve()
    if (config.parent != SLIDE.resolve() or not config.name.startswith("Config_")
            or config.suffix != ".csv" or not config.is_file()):
        fail(f"Expected an existing SLIDE/Config_*.csv: {config}")
    values, original_text = read_config(config)
    overrides = read_overrides(args.settings)
    values.update(overrides)
    inputs, weights = validate(values)
    binary = BIN / "runme"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        fail("Missing executable bin/runme. Build TENNOR into this component's bin directory first.")
    threat_num = integer(values, "threatNum", default=1)
    threads, thread_source = resolve_threads(args.threads, threat_num)
    # Keep automatic names usable for existing filenames containing spaces.
    default_name = re.sub(r"[^A-Za-z0-9_.-]", "_", config.stem.removeprefix("Config_"))
    default_name = default_name.strip("._-")[:80] or "tennor"
    name = args.run_name or (default_name + "_" +
                            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
        fail("--run-name must be 1-120 ASCII letters, digits, dots, underscores or hyphens; start with a letter/digit.")
    runs = (DATASET / "runs").resolve()
    if not contained(runs, DATASET.resolve()):
        fail("dataset/runs must remain inside this component's dataset directory.")
    run_dir = runs / name
    effective_config = SLIDE / (config.stem.split("__run_", 1)[0] + "__run_" + name + ".csv")
    if (run_dir.exists() or run_dir.is_symlink() or
            effective_config.exists() or effective_config.is_symlink()):
        fail("This run name already exists. Choose a new --run-name; outputs are never reused.")
    if "#" in str(ROOT) or "\n" in str(ROOT) or "\r" in str(ROOT):
        fail("The legacy config parser cannot handle '#' or newlines in the component path.")
    effective, outputs, notices = prepare_effective(values, overrides, inputs, weights, run_dir)
    if threads is not None and threat_num > threads:
        notices.append(f"threatNum={threat_num} exceeds OpenMP threads={threads}; "
                       "TENNOR's explicitly sized parallel regions will be capped by OpenMP.")
    if any(character in value for value in effective.values() for character in ("#", "\n", "\r", "\0")):
        fail("The generated config cannot contain '#', newlines, or NUL characters in values.")
    # The C++ parser allocates array fields using the numLayer value seen so far.
    config_text = "# Generated by run_with_set_tennor.py; defaults plus explicit --set overrides.\n"
    config_text += f"numLayer = {effective['numLayer']}\n"
    config_text += "".join(f"{key} = {value}\n" for key, value in effective.items() if key != "numLayer")
    thread_label = str(threads) if threads is not None else "automatic (OpenMP runtime default)"
    print(f"Config:  {config}\nBinary:  {binary}\n"
          f"OpenMP threads: {thread_label} (source: {thread_source})\n"
          f"threatNum: {threat_num}\nOutput:  {run_dir}\n"
          f"Effective config: {effective_config}", flush=True)
    for key, value in overrides.items():
        print(f"Override: {key} = {value}", flush=True)
    for notice in notices:
        print(f"Note: {notice}", flush=True)
    print("Preflight checks data headers only. Data contents and results require manual verification.", flush=True)
    if args.check:
        print("\nEffective configuration preview:\n" + config_text, flush=True)
        print("Preflight passed; no files written and no program launched.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    work = run_dir / "work"
    work.mkdir()
    # TENNOR writes ../dataset/binmaps/* without creating that directory itself.
    (run_dir / "dataset" / "binmaps").mkdir(parents=True)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with effective_config.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(config_text)
    env = os.environ.copy()
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_DYNAMIC", "FALSE")
    command = [str(binary), str(effective_config)]
    metadata = {"component": "TENNOR", "status": "prepared", "config": str(config),
                "config_sha256": digest(config),
                "original_config_text": original_text, "effective_config": str(effective_config),
                "effective_config_sha256": digest(effective_config), "binary": str(binary),
                "binary_sha256": digest(binary), "command": command, "cwd": str(work),
                "inputs": inputs, "effective_values": effective,
                "set_arguments": args.settings, "overrides": overrides, "notices": notices,
                "thread_settings": {"threatNum": threat_num, "omp_num_threads": threads,
                                    "source": thread_source},
                "openmp": {k: v for k, v in env.items() if k.startswith("OMP_")},
                "result_validation": "manual; a zero exit code is not a scientific-result check"}
    for filename in ("build-info.txt", "cnpy.commit"):
        record = BIN / filename
        if record.is_file():
            metadata[filename] = record.read_text(encoding="utf-8")
    manifest = run_dir / "run.json"

    def save_metadata():
        manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    child = None
    interrupted = False
    interrupted_signal = signal.SIGINT

    def handle_termination(signum, frame):
        nonlocal interrupted_signal
        interrupted_signal = signum
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
    start = time.monotonic()
    metadata.update(status="running", started_at=timestamp())
    save_metadata()
    try:
        with (run_dir / "console.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    chunk = child.stdout.read1(65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                returncode = child.wait()
            finally:
                child.stdout.close()
        metadata["status"] = "completed" if returncode == 0 else "failed"
    except KeyboardInterrupt:
        interrupted = True
        metadata.update(status="interrupted", interruption_signal=int(interrupted_signal))
        # Let cleanup finish if a second termination request arrives.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        print(f"Run failed: {error}", file=sys.stderr)
    finally:
        if child is not None:
            stop_child(child)
        metadata.update(finished_at=timestamp(), wall_seconds=time.monotonic() - start,
                        returncode=child.returncode if child is not None else None)
        save_metadata()
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(f"\nRun status: {metadata['status']}; outputs: {run_dir}")
    if interrupted:
        return 128 + interrupted_signal
    code = metadata["returncode"]
    if metadata["status"] != "completed":
        return (128 - code if code is not None and code < 0 else code) or 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
