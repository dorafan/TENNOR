#!/usr/bin/env python3
"""Run a SLIDE+ORAM configuration with repeatable --set KEY=VALUE overrides."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
SLIDE = ROOT / "SLIDE"
DATASET = ROOT / "dataset"
BIN = ROOT / "bin"
KEYS = set("RangePow K L Sparsity Batchsize Rehash Rebuild InputDim totRecords "
           "totRecordsTest Epoch Lr Stepsize numLayer logFile sizesOfLayers "
           "trainData testData weight savedweight".split())
# Existing CSVs may contain this field, but the ORAM C++ parser does not read it.
LEGACY_KEYS = {"retrivalfile"}
COMPILE_TIME_KEYS = {"BUCKETSIZE", "binsize", "LOADWEIGHT", "LoadWeight",
                     "ORAM_N", "SERIAL_LEN", "ADAM", "HashFunction", "Mode"}
BUILD_INFO_FORMAT = "SLIDE_ORAM_BUILD_INFO_V1"
BUILD_INFO_TIMEOUT = 10
CAPACITY_PRESETS = ((65536, "wiki10"), (524288, "amzprime"), (2097152, "wiki325"))


def fail(message):
    raise ValueError(message)


def contained(path, parent):
    return path == parent or parent in path.parents


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_console_line(log, line):
    """Timestamp a received line in local time and tee it to file and terminal."""
    prefix = dt.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ").encode("ascii")
    stamped = prefix + line
    if not stamped.endswith(b"\n"):
        stamped += b"\n"  # Also separate an unterminated final line from our summary.
    log.write(stamped)
    log.flush()
    sys.stdout.buffer.write(stamped)
    sys.stdout.buffer.flush()


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
        if key in COMPILE_TIME_KEYS:
            fail(f"{key} is not a runtime config option in SLIDE+ORAM. "
                 "Change the corresponding macro in SLIDE/Config.h and rebuild; "
                 "the weight-loading macro is named LOADWEIGHT.")
        if key == "retrivalfile":
            fail("SLIDE+ORAM does not read retrivalfile or write that retrieval log.")
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
    for layer in range(1, layers):
        if arrays["L"][layer] > (1 << arrays["RangePow"][layer]):
            fail(f"Layer {layer}: L must be <= 2**RangePow for the ORAM bucket indexing.")
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
    integer(values, "Epoch", minimum=1)
    integer(values, "Stepsize", minimum=1)
    for key in ("Rehash", "Rebuild"):
        integer(values, key, minimum=batch)
    train = source_path(values["trainData"])
    test = source_path(values["testData"])
    if train.parent != test.parent:
        fail("train.txt and test.txt must be in the same dataset directory.")
    inputs = [check_data(train, "train.txt", train_count // batch * batch,
                         features, arrays["sizesOfLayers"][-1]),
              check_data(test, "test.txt", test_count // batch * batch,
                         features, arrays["sizesOfLayers"][-1])]
    weights = None
    if "weight" in values:
        weights = source_path(values["weight"])
        if not contained(weights, DATASET.resolve()):
            fail(f"Checkpoint path must be within this component's dataset: {weights}")
        if weights.exists() and not weights.is_file():
            fail(f"Checkpoint path is not a file: {weights}")
        # LOADWEIGHT is compiled into runme, not read from the CSV. A missing
        # checkpoint is valid for LOADWEIGHT=0; do not infer the binary's setting.
    return inputs, weights


def oram_requirement(values):
    """Budget nodes and hash buckets for the layers written by Network.cpp."""
    sizes = [int(x) for x in values["sizesOfLayers"].split(",")]
    tables = [int(x) for x in values["L"].split(",")]
    powers = [int(x) for x in values["RangePow"].split(",")]
    # C++ stores Sparsity as float, including values rounded to exactly 1.0f.
    sparsity = [struct.unpack("f", struct.pack("f", float(x)))[0]
                for x in values["Sparsity"].split(",")]
    layers = []
    for index in range(int(values["numLayer"])):
        if sparsity[index] != 1.0:
            buckets = tables[index] * (1 << powers[index])
            layers.append({"layer_index": index, "nodes": sizes[index],
                           "hash_buckets": buckets, "logical_blocks": sizes[index] + buckets})
    required = sum(layer["logical_blocks"] for layer in layers)
    return {"required_blocks": required,
            "minimum_oram_n": 1 << (max(1, required) - 1).bit_length(),
            "layers": layers}


def capacity_build_hint(required):
    for capacity, dataset in CAPACITY_PRESETS:
        if capacity >= required:
            return f"bash build.sh --dataset {dataset} (ORAM_N={capacity})"
    minimum = 1 << (required - 1).bit_length()
    return (f"cmake -S . -B bin -DCMAKE_BUILD_TYPE=Debug -DSLIDE_ORAM_N={minimum}, "
            "then cmake --build bin --parallel 4 --target runme")


def has_build_info_marker(binary):
    """Do not invoke older binaries: they treat unknown options as config paths."""
    marker = BUILD_INFO_FORMAT.encode("ascii")
    tail = b""
    with binary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            data = tail + chunk
            if marker in data:
                return True
            tail = data[-(len(marker) - 1):]
    return False


def query_build_info(binary, rebuild_hint):
    advice = f"Rebuild from SLIDE-ORAM/ with {rebuild_hint}."
    if not has_build_info_marker(binary):
        fail(f"{binary} does not advertise the --build-info interface; "
             f"the old binary was not executed. {advice}")
    try:
        result = subprocess.run([str(binary), "--build-info"], cwd=BIN,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=BUILD_INFO_TIMEOUT)
    except subprocess.TimeoutExpired:
        fail(f"bin/runme --build-info timed out after {BUILD_INFO_TIMEOUT} seconds. {advice}")
    except OSError as error:
        fail(f"Cannot query bin/runme --build-info: {error}. {advice}")
    if result.returncode != 0:
        fail(f"bin/runme --build-info failed with exit code {result.returncode}. {advice}")
    try:
        info = json.loads(result.stdout)
    except (ValueError, UnicodeDecodeError):
        fail(f"bin/runme --build-info did not return valid JSON. {advice}")
    if (not isinstance(info, dict) or info.get("format") != BUILD_INFO_FORMAT
            or type(info.get("schema_version")) is not int or info["schema_version"] != 1):
        fail(f"bin/runme --build-info returned an unsupported format or schema version. {advice}")
    for key in ("ORAM_N", "SERIAL_LEN"):
        if type(info.get(key)) is not int or info[key] < 1:
            fail(f"bin/runme --build-info must report a positive integer {key}. {advice}")
    capacity = info["ORAM_N"]
    if capacity & (capacity - 1):
        fail(f"Compiled ORAM_N={capacity} is not a power of two. {advice}")
    return info


def check_compiled_capacity(values, binary):
    requirement = oram_requirement(values)
    hint = capacity_build_hint(requirement["required_blocks"])
    info = query_build_info(binary, hint)
    required = requirement["required_blocks"]
    capacity = info["ORAM_N"]
    if capacity < required:
        fail(f"Compiled ORAM_N={capacity}, but this configuration requires {required} logical "
             f"blocks (minimum power-of-two capacity {requirement['minimum_oram_n']}). "
             f"Rebuild from SLIDE-ORAM/ with {hint}.")
    requirement["compiled_oram_n"] = capacity
    requirement["serialization_capacity_checked"] = False
    return info, requirement


def prepare_effective(values, overrides, inputs, weights, run_dir):
    effective = dict(values)
    legacy_retrieval = effective.pop("retrivalfile", None)
    effective.update(trainData=inputs[0]["path"], testData=inputs[1]["path"])
    outputs = {}
    defaults = {"logFile": "training.log", "savedweight": "savedWeights.npz"}
    for key, default in defaults.items():
        path = Path(overrides.get(key, default))
        path = (path if path.is_absolute() else run_dir / path).resolve()
        if not contained(path, run_dir) or path == run_dir:
            fail(f"--set {key}: output must be a file inside this run directory: {run_dir}")
        if contained(path, run_dir / "work"):
            fail(f"--set {key}: work/ is reserved for the program's working directory.")
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
    notices = ["The supplied ORAM source has its evaluation calls commented out; "
               "this launcher does not enable evaluation.",
               "ORAM_N is checked against the executable's reported compile-time capacity. "
               "SERIAL_LEN is reported but its adequacy is not checked. "
               "Changing Batchsize or layer sizes requires checking serialization capacity; "
               "see README.md."]
    if legacy_retrieval is not None:
        notices.append("Ignored legacy retrivalfile from the original CSV: SLIDE+ORAM does not "
                       "read this field, so it is omitted from the generated config.")
    if weights is None or not weights.is_file():
        notices.append("No existing checkpoint was found for weight. This is valid only for "
                       "a runme built with LOADWEIGHT=0; LOADWEIGHT=1 requires an existing "
                       "compatible checkpoint. The launcher does not inspect that build setting.")
    return effective, outputs, notices


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
    parser.add_argument("--check", action="store_true",
                        help="check config/headers and query compiled capacity; no run files or training")
    parser.add_argument("--threads", type=int, help="OpenMP threads (default: OMP_NUM_THREADS or 1)")
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
        fail("Missing executable bin/runme. Build SLIDE+ORAM into this component's bin directory first.")
    build_info, capacity_check = check_compiled_capacity(values, binary)
    stdbuf = shutil.which("stdbuf")
    if stdbuf is None:
        fail("Missing stdbuf (GNU coreutils). Run bash ../install_dependencies.sh first.")
    stdbuf = str(Path(stdbuf).resolve())
    try:
        threads = args.threads if args.threads is not None else int(os.environ.get("OMP_NUM_THREADS", "1"))
    except ValueError:
        fail("OMP_NUM_THREADS must be a positive integer, or supply --threads.")
    if threads < 1:
        fail("--threads must be positive.")
    default_name = re.sub(r"[^A-Za-z0-9_.-]", "_", config.stem.removeprefix("Config_"))
    default_name = default_name.strip("._-")[:80] or "oram"
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
    if any(character in value for value in effective.values() for character in ("#", "\n", "\r", "\0")):
        fail("The generated config cannot contain '#', newlines, or NUL characters in values.")
    # The C++ parser allocates array fields using the numLayer value seen so far.
    config_text = "# Generated by run_with_set_oram.py; defaults plus explicit --set overrides.\n"
    config_text += f"numLayer = {effective['numLayer']}\n"
    config_text += "".join(f"{key} = {value}\n" for key, value in effective.items() if key != "numLayer")
    print(f"Config:  {config}\nBinary:  {binary}\nOMP_NUM_THREADS: {threads}\nOutput:  {run_dir}\n"
          f"Effective config: {effective_config}", flush=True)
    print(f"ORAM capacity: compiled={build_info['ORAM_N']}, "
          f"required={capacity_check['required_blocks']}, "
          f"minimum={capacity_check['minimum_oram_n']}; "
          f"compiled SERIAL_LEN={build_info['SERIAL_LEN']} (adequacy not checked)", flush=True)
    for key, value in overrides.items():
        print(f"Override: {key} = {value}", flush=True)
    for notice in notices:
        print(f"Note: {notice}", flush=True)
    print("Preflight checks config, paths, data headers, and actual compiled ORAM capacity; "
          "serialization capacity, checkpoints, and scientific results require manual verification.",
          flush=True)
    if args.check:
        print("\nEffective configuration preview:\n" + config_text, flush=True)
        print("Preflight passed; only the read-only build-info query was executed. "
              "No run files written and no training launched.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    work = run_dir / "work"
    work.mkdir()
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with effective_config.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(config_text)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_DYNAMIC", "FALSE")
    command = [stdbuf, "-oL", str(binary), str(effective_config)]
    metadata = {"component": "SLIDE+ORAM", "status": "prepared", "config": str(config),
                "config_sha256": digest(config),
                "original_config_text": original_text, "effective_config": str(effective_config),
                "effective_config_sha256": digest(effective_config), "binary": str(binary),
                "binary_sha256": digest(binary), "command": command, "cwd": str(work),
                "binary_build_info": build_info, "oram_capacity": capacity_check,
                "inputs": inputs, "effective_values": effective,
                "set_arguments": args.settings, "overrides": overrides, "notices": notices,
                "weight_input": {"path": str(weights) if weights is not None else None,
                                 "exists": weights.is_file() if weights is not None else False,
                                 "loadweight_build_setting": "not inspected"},
                "preflight_scope": "config/paths/data headers/actual compiled ORAM capacity; "
                                   "serialization adequacy and checkpoints not checked",
                "console_logging": {"timestamps": "local time at line receipt",
                                    "format": "[%Y-%m-%d %H:%M:%S]",
                                    "stdout_buffering": "stdbuf -oL"},
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

    with (run_dir / "console.log").open("xb") as log:
        previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
        start = time.monotonic()
        metadata.update(status="running", started_at=timestamp())
        try:
            save_metadata()
            write_console_line(log, b"[launcher] Starting runme")
            child = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                for line in iter(child.stdout.readline, b""):
                    write_console_line(log, line)
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
        finally:
            try:
                if child is not None:
                    stop_child(child)
                metadata.update(finished_at=timestamp(), wall_seconds=time.monotonic() - start,
                                returncode=child.returncode if child is not None else None)
                save_metadata()
                if "error" in metadata:
                    write_console_line(log, f"[launcher] Run failed: {metadata['error']}".encode("utf-8"))
                summary = (f"[launcher] Run status: {metadata['status']}; "
                           f"returncode: {metadata['returncode']}; "
                           f"wall time: {metadata['wall_seconds']:.3f} seconds")
                write_console_line(log, summary.encode("utf-8"))
                write_console_line(log, f"[launcher] Outputs: {run_dir}".encode("utf-8"))
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
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
