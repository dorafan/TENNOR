"""Synthetic-log checks for independent WTA / MP-WTA accuracy comparisons.

These tests never invoke a training executable or read training datasets.
"""

import csv
import hashlib
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import analysis_common as base
import analyze_slide_mp as analyzer
import wta_logs


class EpochBoundaryTests(unittest.TestCase):
    def test_final_evaluation_is_retained_without_a_launcher_manifest(self):
        rows, audit = base.parse_log(
            "Epoch 0\n100 3 0.9\n442 20 0.2\n"
            "Epoch 1\n800 40 0.8\n884 45 0.4\n",
            epoch_batches=442, configured_epochs=2,
        )
        self.assertEqual([row["epoch"] for row in rows], [0, 1])
        self.assertEqual([row["p_at_1"] for row in rows], [0.2, 0.4])
        self.assertEqual(audit[-1]["reason"], "epoch_end_iteration")

    def test_later_header_does_not_confirm_a_mid_epoch_evaluation(self):
        rows, _ = base.parse_log(
            "Epoch 0\n400 20 0.9\nEpoch 1\n884 45 0.4\n",
            epoch_batches=442, configured_epochs=2,
        )
        self.assertEqual([row["epoch"] for row in rows], [1])

    def test_bad_tail_never_falls_back_to_an_earlier_evaluation(self):
        for tail in ("884 45", "884 45 nan", "884 45 1.2", "800 40 0.8"):
            with self.subTest(tail=tail):
                rows, audit = base.parse_log(
                    "Epoch 0\n442 20 0.2\nEpoch 1\n700 30 0.6\n" + tail + "\n",
                    epoch_batches=442, configured_epochs=2,
                )
                self.assertEqual([row["epoch"] for row in rows], [0])
                self.assertEqual(audit[-1]["disposition"], "discarded")

    def test_epoch_outside_the_declared_training_limit_is_not_accepted(self):
        rows, _ = base.parse_log(
            "Epoch 2\n1326 70 0.9\n", epoch_batches=442, configured_epochs=2,
        )
        self.assertEqual(rows, [])

    def test_legacy_mp_completion_behavior_is_preserved(self):
        text = "Epoch 0\n442 20 0.2\nEpoch 1\n884 45 0.4\n"
        unconfirmed, _ = base.parse_log(text)
        confirmed, _ = base.parse_log(text, completed_epoch=1)
        self.assertEqual([row["epoch"] for row in unconfirmed], [0])
        self.assertEqual([row["epoch"] for row in confirmed], [0, 1])


class ArtifactFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="wta-analysis-test-", dir=SCRIPTS.parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assertIn(SCRIPTS.parent.resolve(), self.root.parents)
        (self.root / "scripts").mkdir()
        for module in (base, analyzer, wta_logs):
            copied = self.root / "scripts" / Path(module.__file__).name
            copied.write_bytes(Path(module.__file__).read_bytes())
            file_patch = patch.object(module, "__file__", str(copied))
            file_patch.start()
            self.addCleanup(file_patch.stop)
        root_patch = patch.object(base, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.config = self.root / "SLIDE/SLIDE/Config_wiki10_wta.csv"
        self.log = self.root / "SLIDE/dataset/Wiki10/training_wta.log"
        self.values = {
            "numLayer": "2", "RangePow": "6,9", "K": "2,3", "L": "1,50",
            "Sparsity": "1,0.005,1,1", "Batchsize": "32", "Rehash": "320",
            "Rebuild": "3200", "InputDim": "101938", "totRecords": "14146",
            "totRecordsTest": "6616", "Lr": "0.0001", "Epoch": "2",
            "Stepsize": "100", "sizesOfLayers": "256,30938",
            "trainData": "../dataset/Wiki10/train.txt",
            "testData": "../dataset/Wiki10/test.txt",
            "weight": "../dataset/Wiki10/savedWeights_wta.npz",
            "savedweight": "../dataset/Wiki10/savedWeights_wta.npz",
            "logFile": "../dataset/Wiki10/training_wta.log",
        }
        self.write_config(self.config)
        self.write(self.log, "Epoch 0\n100 3 0.9\n442 20 0.2\nEpoch 1\n884 45 0.4\n")

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_config(self, path, **overrides):
        values = dict(self.values, **overrides)
        return self.write(path, "# Saved configuration for synthetic logs\n\n" + "\n".join(
            f"{key} = {value}" for key, value in values.items()) + "\n")

    def make_mp(self):
        directory = self.root / "SLIDE-MP/dataset/runs/wiki10-repeat1"
        log = self.write(directory / "training.log",
                         "Epoch 0\n442 18 0.3\nEpoch 1\n884 41 0.6\n")
        values = dict(self.values, L="1,1", isMPWTA="1", isBruteForce="0",
                      BUCKETSIZE="512", binsize="8", logFile="training.log")
        self.write(directory / "run.json", json.dumps({
            "status": "completed", "returncode": 0, "effective_values": values,
        }))
        return log

    def analyze(self, *options):
        with patch.object(base, "make_plot", return_value="synthetic-test"), redirect_stdout(io.StringIO()):
            return analyzer.main(["--dataset", "wiki10", "--run", "wiki10-repeat1", *options])

    def read_csv(self, filename):
        with (self.root / "analysis/slide-mp/wiki10" / filename).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


class WtaLoadingTests(ArtifactFixture):
    def test_default_log_and_csv_are_loaded_and_hashed_without_dataset_access(self):
        runs = wta_logs.load_wta_runs([], [], "wiki10")
        self.assertEqual(len(runs), 1)
        self.assertEqual([row["epoch"] for row in runs[0]["rows"]], [0, 1])
        self.assertEqual(runs[0]["log_sha256"], hashlib.sha256(self.log.read_bytes()).hexdigest())
        self.assertEqual(runs[0]["config_sha256"], hashlib.sha256(self.config.read_bytes()).hexdigest())
        self.assertEqual(runs[0]["evaluation_policy"], "full")
        self.assertEqual(runs[0]["rows"][0]["test_records"], 6592)
        self.assertEqual(runs[0]["rows"][1]["test_records"], 6592)
        self.assertTrue(all(row["evaluation_scope"] == "all_complete_test_batches"
                            for row in runs[0]["rows"]))
        self.assertFalse((self.log.parent / "train.txt").exists())

    def test_legacy_evaluation_requires_explicit_selection(self):
        run = wta_logs.load_wta_runs([], [], "wiki10", evaluation="legacy50")[0]
        self.assertEqual(run["evaluation_policy"], "legacy50")
        self.assertEqual([row["test_records"] for row in run["rows"]], [1600, 6592])
        self.assertEqual([row["evaluation_scope"] for row in run["rows"]],
                         ["first_50_test_batches", "all_complete_test_batches"])

    def test_full_evaluation_supports_fewer_than_fifty_test_batches(self):
        self.write_config(self.config, totRecordsTest="1000")
        run = wta_logs.load_wta_runs([], [], "wiki10")[0]
        self.assertEqual([row["test_batches"] for row in run["rows"]], [31, 31])
        self.assertEqual([row["test_records"] for row in run["rows"]], [992, 992])
        with self.assertRaises(ValueError):
            wta_logs.load_wta_runs([], [], "wiki10", evaluation="legacy50")

    def test_independent_repetitions_get_equal_weight_and_sample_sd(self):
        second = self.write(self.log.parent / "repeat2.log",
                            "Epoch 0\n442 23 0.6\nEpoch 1\n884 48 0.8\n")
        runs = wta_logs.load_wta_runs([self.log, second], [self.config], "wiki10")
        summary = wta_logs.summarize_wta(runs)
        self.assertEqual([row["n_runs"] for row in summary], [2, 2])
        self.assertAlmostEqual(summary[0]["p_at_1_mean"], 0.4)
        self.assertAlmostEqual(summary[1]["p_at_1_mean"], 0.6)
        self.assertAlmostEqual(summary[0]["p_at_1_std"], math.sqrt(0.08))
        self.assertEqual(summary[0]["test_records"], 6592)
        self.assertEqual(summary[1]["test_records"], 6592)

    def test_missing_epochs_are_not_filled(self):
        second = self.write(self.log.parent / "repeat2.log", "Epoch 1\n884 48 0.8\n")
        rows = wta_logs.summarize_wta(wta_logs.load_wta_runs([self.log, second], [], "wiki10"))
        self.assertEqual([row["n_runs"] for row in rows], [1, 2])
        self.assertIsNone(rows[0]["p_at_1_std"])
        self.assertAlmostEqual(rows[1]["p_at_1_mean"], 0.6)

    def test_repeated_paths_and_copied_logs_cannot_be_counted_as_repetitions(self):
        copied = self.write(self.log.parent / "copy.log", self.log.read_text())
        for pair in ([self.log, self.log], [self.log, copied]):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                wta_logs.load_wta_runs(pair, [], "wiki10")

    def test_incompatible_configs_and_wrong_pair_counts_are_rejected(self):
        second = self.write(self.log.parent / "repeat2.log", "Epoch 1\n884 48 0.8\n")
        other = self.write_config(self.config.parent / "config_other.csv", K="2,4")
        with self.assertRaises(ValueError):
            wta_logs.load_wta_runs([self.log, second], [self.config, other], "wiki10")
        with self.assertRaises(ValueError):
            wta_logs.load_wta_runs([self.log], [self.config, other], "wiki10")

    def test_dataset_conflicts_are_rejected(self):
        self.write_config(self.config, trainData="../dataset/Amazon/train.txt")
        with self.assertRaises(ValueError):
            wta_logs.load_wta_runs([], [], "wiki10")

    def test_invalid_declared_dimensions_and_hash_counts_are_rejected(self):
        for key, value in (("Batchsize", "0"), ("Epoch", "-1"),
                           ("L", "1,1"), ("sizesOfLayers", "256")):
            with self.subTest(key=key, value=value):
                self.write_config(self.config, **{key: value})
                with self.assertRaises(ValueError):
                    wta_logs.load_wta_runs([], [], "wiki10")

    def test_log_path_cannot_escape_artifact(self):
        with self.assertRaises(ValueError):
            wta_logs.load_wta_runs([SCRIPTS / "analysis_common.py"], [], "wiki10")


class ComparisonCommandTests(ArtifactFixture):
    def setUp(self):
        super().setUp()
        self.make_mp()

    def test_measured_logs_replace_history_and_difference_requires_matching_test_scope(self):
        self.assertEqual(self.analyze("--historical-wta"), 0)
        self.assertEqual(len(self.read_csv("wta_reference.csv")), 30)
        self.assertEqual(self.analyze(), 0)
        rows = self.read_csv("comparison.csv")
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]["wta_p_at_1_mean"]), 0.2)
        self.assertAlmostEqual(float(rows[1]["wta_p_at_1_mean"]), 0.4)
        self.assertAlmostEqual(float(rows[0]["p_at_1_difference"]), 0.1)
        self.assertAlmostEqual(float(rows[1]["p_at_1_difference"]), 0.2)
        self.assertEqual(rows[0]["baseline_kind"], "measured_logs")
        self.assertEqual(int(rows[0]["wta_test_records"]), 6592)
        self.assertEqual(int(rows[1]["wta_test_records"]), 6592)
        self.assertEqual(int(rows[1]["mp_test_records"]), 6592)
        self.assertEqual(len(self.read_csv("wta_epochs.csv")), 2)
        self.assertEqual(len(self.read_csv("wta_summary.csv")), 2)
        self.assertEqual(self.read_csv("wta_reference.csv"), [])
        report = json.loads((self.root / "analysis/slide-mp/wiki10/analysis.json").read_text())
        serialized = json.dumps(report)
        self.assertIn(hashlib.sha256(self.log.read_bytes()).hexdigest(), serialized)
        self.assertIn(hashlib.sha256(self.config.read_bytes()).hexdigest(), serialized)
        self.assertIn("epoch_end_iteration", serialized)
        self.assertEqual(report["wta_runs"][0]["evaluation_policy"], "full")

    def test_legacy_logs_keep_partial_coverage_and_exclude_unmatched_differences(self):
        self.assertEqual(self.analyze("--wta-evaluation", "legacy50"), 0)
        rows = self.read_csv("comparison.csv")
        self.assertEqual(rows[0]["evaluation_match"], "False")
        self.assertEqual(rows[0]["p_at_1_difference"], "")
        self.assertEqual(int(rows[0]["wta_test_records"]), 1600)
        self.assertEqual(rows[1]["evaluation_match"], "True")
        self.assertAlmostEqual(float(rows[1]["p_at_1_difference"]), 0.2)
        report = json.loads((self.root / "analysis/slide-mp/wiki10/analysis.json").read_text())
        self.assertEqual(report["wta_runs"][0]["evaluation_policy"], "legacy50")

    def test_full_evaluations_with_different_sample_counts_do_not_get_a_difference(self):
        self.write_config(self.config, totRecordsTest="6400")
        self.assertEqual(self.analyze(), 0)
        for row in self.read_csv("comparison.csv"):
            self.assertEqual(row["evaluation_match"], "False")
            self.assertEqual(row["p_at_1_difference"], "")

    def test_explicit_evaluation_policy_cannot_be_combined_with_history(self):
        for policy in ("full", "legacy50"):
            with self.subTest(policy=policy), self.assertRaises((ValueError, SystemExit)):
                self.analyze("--historical-wta", "--wta-evaluation", policy)
        self.assertFalse((self.root / "analysis").exists())

    def test_missing_log_fails_without_silent_historical_fallback(self):
        self.log.unlink()
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "historical-wta"):
            self.analyze()
        self.assertFalse((self.root / "analysis").exists())

    def test_list_does_not_require_wta_results(self):
        self.log.unlink()
        self.assertEqual(self.analyze("--list"), 0)
        self.assertFalse((self.root / "analysis").exists())

    def test_history_must_be_explicit_and_does_not_claim_measured_run_counts(self):
        self.assertEqual(self.analyze(), 0)
        self.assertTrue(self.read_csv("wta_epochs.csv"))
        self.log.unlink()
        self.assertEqual(self.analyze("--historical-wta"), 0)
        reference = self.read_csv("wta_reference.csv")
        self.assertEqual(len(reference), 30)
        self.assertAlmostEqual(float(reference[0]["p_at_1"]), analyzer.WTA_REFERENCE["wiki10"][0][1])
        self.assertEqual(self.read_csv("wta_epochs.csv"), [])
        self.assertEqual(self.read_csv("wta_summary.csv"), [])
        for row in self.read_csv("comparison.csv"):
            self.assertEqual(row["baseline_kind"], "historical")
            self.assertEqual(row["wta_n_runs"], "")
            self.assertEqual(row["p_at_1_difference"], "")

    def test_output_cannot_overwrite_slide_inputs(self):
        with self.assertRaises(ValueError):
            self.analyze("--output-dir", "SLIDE/results")
        self.assertFalse((self.root / "SLIDE/results").exists())


if __name__ == "__main__":
    unittest.main()
