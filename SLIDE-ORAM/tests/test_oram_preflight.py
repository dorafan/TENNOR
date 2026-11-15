"""Capacity-preflight regressions; all training subprocesses are mocked."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


COMPONENT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oram_launcher", COMPONENT / "run_with_set_oram.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def report(capacity=65536, **changes):
    info = {"format": launcher.BUILD_INFO_FORMAT, "schema_version": 1,
            "ORAM_N": capacity, "SERIAL_LEN": 6144}
    info.update(changes)
    return info


def response(info=None, output=None, returncode=0):
    if output is None:
        output = json.dumps(info if info is not None else report()).encode("utf-8")
    return subprocess.CompletedProcess([], returncode, output, b"")


class CapacityPreflightTests(unittest.TestCase):
    def setUp(self):
        # Keep even transient test outputs within this review copy.
        self.temp = tempfile.TemporaryDirectory(prefix=".preflight-test-", dir=COMPONENT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.slide = self.root / "SLIDE"
        self.data = self.root / "dataset"
        self.bin = self.root / "bin"
        for directory in (self.slide, self.data / "Wiki10", self.bin):
            directory.mkdir(parents=True)
        self.binary = self.bin / "runme"
        self.binary.write_bytes(launcher.BUILD_INFO_FORMAT.encode("ascii"))
        self.binary.chmod(0o755)
        for name, value in (("ROOT", self.root), ("SLIDE", self.slide),
                            ("DATASET", self.data), ("BIN", self.bin)):
            patch = mock.patch.object(launcher, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.values, _ = launcher.read_config(COMPONENT / "SLIDE/Config_wiki10_batch32.csv")
        labels = self.values["sizesOfLayers"].split(",")[-1]
        for key, filename, count_key in (("trainData", "train.txt", "totRecords"),
                                         ("testData", "test.txt", "totRecordsTest")):
            path = self.data / "Wiki10" / filename
            path.write_text(f"{self.values[count_key]} {self.values['InputDim']} {labels}\n",
                            encoding="utf-8")
            self.values[key] = str(path)
        self.config = self.slide / "Config_wiki10_batch32.csv"
        self.config.write_text("".join(f"{k}={v}\n" for k, v in self.values.items()),
                               encoding="utf-8")

    def files(self):
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))

    def main_context(self, args):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(sys, "argv", ["run_with_set_oram.py",
                            self.config.name, "--run-name", "capacity-repeat1"] + args))
        stack.enter_context(mock.patch.object(sys, "platform", "linux"))
        stack.enter_context(mock.patch.object(launcher.shutil, "which", return_value=str(self.bin / "stdbuf")))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_supplied_capacity_requirements(self):
        for dataset, required, minimum in (("wiki10", 56538, 65536),
                                           ("amzprime", 297614, 524288),
                                           ("wiki325", 1963456, 2097152)):
            with self.subTest(dataset=dataset):
                values, _ = launcher.read_config(COMPONENT / "SLIDE" / f"Config_{dataset}_batch32.csv")
                actual = launcher.oram_requirement(values)
                self.assertEqual(actual["required_blocks"], required)
                self.assertEqual(actual["minimum_oram_n"], minimum)

    def test_training_sparsity_uses_float32_and_ignores_evaluation_entries(self):
        self.values["Sparsity"] = "1,0.99999999,0.1,0.1"
        self.assertEqual(launcher.oram_requirement(self.values)["required_blocks"], 0)
        self.values["Sparsity"] = "1,0.999,1,1"
        self.assertEqual(launcher.oram_requirement(self.values)["required_blocks"], 56538)

    def test_old_binary_is_not_executed(self):
        self.binary.write_bytes(b"legacy binary without query support")
        with mock.patch.object(launcher.subprocess, "run") as query:
            with self.assertRaisesRegex(ValueError, "old binary was not executed"):
                launcher.check_compiled_capacity(self.values, self.binary)
            query.assert_not_called()

    def test_marker_across_read_boundary(self):
        self.binary.write_bytes(b"x" * (1024 * 1024 - 5)
                                + launcher.BUILD_INFO_FORMAT.encode("ascii") + b"x")
        self.assertTrue(launcher.has_build_info_marker(self.binary))

    def test_query_uses_binary_not_header(self):
        (self.slide / "Config.h").write_text("#define ORAM_N 65536\n", encoding="utf-8")
        values, _ = launcher.read_config(COMPONENT / "SLIDE/Config_amzprime_batch32.csv")
        with mock.patch.object(launcher.subprocess, "run", return_value=response(report(524288))) as query:
            info, capacity = launcher.check_compiled_capacity(values, self.binary)
        self.assertEqual(info["ORAM_N"], 524288)
        self.assertEqual(capacity["required_blocks"], 297614)
        self.assertEqual(query.call_args.args[0], [str(self.binary), "--build-info"])
        self.assertEqual(query.call_args.kwargs["cwd"], self.bin)
        self.assertEqual(query.call_args.kwargs["timeout"], launcher.BUILD_INFO_TIMEOUT)

    def test_new_header_cannot_hide_undersized_binary(self):
        (self.slide / "Config.h").write_text("#define ORAM_N 2097152\n", encoding="utf-8")
        values, _ = launcher.read_config(COMPONENT / "SLIDE/Config_amzprime_batch32.csv")
        with mock.patch.object(launcher.subprocess, "run", return_value=response()):
            with self.assertRaisesRegex(ValueError, "ORAM_N=65536.*297614.*--dataset amzprime"):
                launcher.check_compiled_capacity(values, self.binary)

    def test_invalid_build_info_is_rejected(self):
        invalid = [b"not JSON", b"{}", b"[]", b"\xff",
                   json.dumps(report(format="other")).encode(),
                   json.dumps(report(schema_version=True)).encode(),
                   json.dumps(report(schema_version=2)).encode(),
                   json.dumps(report(ORAM_N=True)).encode(),
                   json.dumps(report(ORAM_N=65536.0)).encode(),
                   json.dumps(report(ORAM_N=0)).encode(),
                   json.dumps(report(ORAM_N=65535)).encode(),
                   json.dumps(report(SERIAL_LEN=False)).encode(),
                   json.dumps(report(SERIAL_LEN=0)).encode()]
        for output in invalid:
            with self.subTest(output=output):
                with mock.patch.object(launcher.subprocess, "run", return_value=response(output=output)):
                    with self.assertRaises(ValueError):
                        launcher.check_compiled_capacity(self.values, self.binary)

    def test_query_failure_and_timeout_are_actionable(self):
        with mock.patch.object(launcher.subprocess, "run", return_value=response(returncode=2)):
            with self.assertRaisesRegex(ValueError, "exit code 2.*Rebuild"):
                launcher.check_compiled_capacity(self.values, self.binary)
        with mock.patch.object(launcher.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("runme", 10)):
            with self.assertRaisesRegex(ValueError, "timed out.*Rebuild"):
                launcher.check_compiled_capacity(self.values, self.binary)

    def test_capacity_failure_blocks_check_and_regular_launch_without_writes(self):
        before = self.files()
        for args in (["--check", "--set", "L=1,200"], ["--set", "L=1,200"]):
            with self.subTest(args=args), self.main_context(args):
                with mock.patch.object(launcher.subprocess, "run", return_value=response()):
                    with mock.patch.object(launcher.subprocess, "Popen") as training:
                        with self.assertRaisesRegex(ValueError, "requires 133338 logical blocks"):
                            launcher.main()
                        training.assert_not_called()
                self.assertEqual(self.files(), before)

    def test_successful_check_is_read_only_and_starts_no_training(self):
        before = self.files()
        with self.main_context(["--check"]):
            with mock.patch.object(launcher.subprocess, "run", return_value=response()):
                with mock.patch.object(launcher.subprocess, "Popen") as training:
                    self.assertEqual(launcher.main(), 0)
                    training.assert_not_called()
        self.assertEqual(self.files(), before)

    def test_normal_launch_records_capacity_with_mocked_child(self):
        child = mock.Mock(stdout=io.BytesIO(b""), returncode=0)
        child.wait.return_value = 0
        child.poll.return_value = 0
        with self.main_context([]):
            with mock.patch.object(launcher.subprocess, "run", return_value=response()):
                with mock.patch.object(launcher.subprocess, "Popen", return_value=child):
                    with mock.patch.object(launcher, "write_console_line"):
                        self.assertEqual(launcher.main(), 0)
        metadata = json.loads((self.data / "runs/capacity-repeat1/run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["binary_build_info"], report())
        self.assertEqual(metadata["oram_capacity"]["required_blocks"], 56538)
        self.assertEqual(metadata["oram_capacity"]["compiled_oram_n"], 65536)
        self.assertFalse(metadata["oram_capacity"]["serialization_capacity_checked"])


if __name__ == "__main__":
    unittest.main()
