"""Regresiones del protocolo: comparabilidad, clases ausentes y rechazo de regresiones."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from utils.config import CLASSES, PROJECT_ROOT
from utils.quality_route import PROTOCOL, classification_metrics, compare_reports, prepare_candidate_config, prepare_experiments, write_new_json


def fixture_report():
    return {"protocol": PROTOCOL, "evidence": "fresh_runtime_inference", "test_used": False,
            "cohort": {"manifest_sha256": "fixture", "objects": 100},
            "environment": {"cpu": "fixture", "threads": 2}, "checkpoint_hashes": {"model": "fixture"},
            "evaluation_code": {"engine": "fixture"},
            "metrics": {
                "original": {"macro_f1": .78, "class_support": dict.fromkeys(CLASSES, 10),
                             "class_recall": dict.fromkeys(CLASSES, .8)},
                "scene": {"accuracy": .44, "macro_f1_present": .34},
                "detector": {"mAP50": .34, "precision": .47, "recall": .34},
                "detector_by_source": {"TACO": {"mAP50_95": .20}, "MJU": {"mAP50_95": .10}},
                "joint": {"recall": .24}, "primary_joint_diagnostic": {"recall": .16},
                "latency": {"p95_ms": 200.}}}


class QualityRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads((PROJECT_ROOT / "config/quality_route.json").read_text(encoding="utf-8"))

    def test_missing_classes_do_not_report_recall_as_zero_or_inflate_present_macro(self):
        metrics = classification_metrics(["plastico", "plastico"], ["plastico", "metal"], CLASSES)
        self.assertEqual(metrics["accuracy"], .5)
        self.assertIsNone(metrics["class_recall"]["organicos"])
        self.assertEqual(metrics["class_support"]["organicos"], 0)
        self.assertAlmostEqual(metrics["macro_f1_present"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3) / 6)

    def test_empty_or_mismatched_classification_is_invalid(self):
        for a, b in (([], []), (["plastico"], [])):
            with self.assertRaises(ValueError):
                classification_metrics(a, b, CLASSES)

    def test_identical_candidate_has_no_claimed_improvement(self):
        a = fixture_report()
        self.assertFalse(compare_reports(a, copy.deepcopy(a), self.policy)["offline_pass"])

    def test_pipeline_gain_passes_only_with_all_guards(self):
        a, b = fixture_report(), fixture_report()
        b["metrics"]["joint"]["recall"] += .04
        b["metrics"]["primary_joint_diagnostic"]["recall"] += .02
        self.assertTrue(compare_reports(a, b, self.policy)["offline_pass"])
        b["metrics"]["detector_by_source"]["MJU"]["mAP50_95"] = .05
        self.assertFalse(compare_reports(a, b, self.policy)["offline_pass"])

    def test_material_gain_cannot_hide_original_class_regression(self):
        a, b = fixture_report(), fixture_report()
        b["metrics"]["scene"].update(accuracy=.52, macro_f1_present=.40)
        self.assertTrue(compare_reports(a, b, self.policy, "material")["offline_pass"])
        b["metrics"]["original"]["class_recall"]["organicos"] = .70
        self.assertFalse(compare_reports(a, b, self.policy, "material")["offline_pass"])

    def test_detector_recall_gain_cannot_hide_false_positives_or_latency(self):
        a, b = fixture_report(), fixture_report()
        b["metrics"]["detector"].update(mAP50=.39, recall=.40)
        self.assertTrue(compare_reports(a, b, self.policy, "detector")["offline_pass"])
        b["metrics"]["detector"]["precision"] = .30
        self.assertFalse(compare_reports(a, b, self.policy, "detector")["offline_pass"])
        b["metrics"]["detector"]["precision"] = .47
        b["metrics"]["latency"]["p95_ms"] = 300
        self.assertFalse(compare_reports(a, b, self.policy, "detector")["offline_pass"])

    def test_incomparable_evidence_is_rejected(self):
        for key, value in (("protocol", "other"), ("evidence", "historical"), ("test_used", True),
                           ("test_used", None), ("cohort", {"other": 1}), ("environment", {"other": 1}),
                           ("checkpoint_hashes", {}), ("evaluation_code", {}), ("evaluation_code", {"engine": "other"})):
            with self.subTest(key=key):
                a, b = fixture_report(), fixture_report()
                b[key] = value
                with self.assertRaises(ValueError):
                    compare_reports(a, b, self.policy)

    def test_nonfinite_or_impossible_metrics_do_not_pass(self):
        for value in (float("nan"), float("inf"), -1, 1.1, True, "0.8", None):
            with self.subTest(value=value):
                a, b = fixture_report(), fixture_report()
                b["metrics"]["original"]["macro_f1"] = value
                with self.assertRaises(ValueError):
                    compare_reports(a, b, self.policy)

    def test_output_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            write_new_json(path, {"first": 1})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"second": 2})
            self.assertEqual(json.loads(path.read_text()), {"first": 1})

    def test_experiments_change_one_factor_and_do_not_replace_runtime(self):
        runtime = (PROJECT_ROOT / "config/app.json").read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "plan"
            plan = prepare_experiments(output, [72])
            self.assertFalse(plan["launched_training"])
            self.assertEqual(len(plan["experiments"]), 4)
            for focus in ("material", "detector"):
                pair = [r for r in plan["experiments"] if r["focus"] == focus]
                a, b = [json.loads(Path(r["config"]).read_text()) for r in pair]
                self.assertEqual([k for k in a if a[k] != b[k]], [pair[0]["tested_variable"]])
                self.assertTrue(all("--device" in r["training_args"] and "cuda" in r["training_args"] for r in pair))
                self.assertFalse((output / "runs").exists())
            with self.assertRaises(FileExistsError):
                prepare_experiments(output, [72])
        self.assertEqual((PROJECT_ROOT / "config/app.json").read_bytes(), runtime)

    def test_invalid_seeds_do_not_create_partial_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "plan"
            for seeds in ([], [1, 1], [-1], [True]):
                with self.assertRaises(ValueError):
                    prepare_experiments(output, seeds)
            self.assertFalse(output.exists())

    def test_material_plan_preserves_explicit_dataset_version(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = prepare_experiments(Path(folder) / 'plan', [72], focuses=('material',),
                                       scenes='data/supplemental/local_version.zip')
            self.assertEqual(len(plan['experiments']), 2)
            self.assertTrue(all(e['focus'] == 'material' and 'data/supplemental/local_version.zip' in e['training_args']
                                for e in plan['experiments']))

    def test_candidate_config_preserves_active_settings_and_other_models(self):
        runtime = (PROJECT_ROOT / "config/app.json").read_bytes()
        before = json.loads(runtime)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "candidate.pth"
            checkpoint.write_bytes(b"fixture-only")
            for focus, key in (("material", "material_candidate_checkpoint"), ("detector", "detector_checkpoint")):
                output = Path(folder) / (focus + ".json")
                report = prepare_candidate_config(checkpoint, output, focus)
                after = json.loads(output.read_text())
                self.assertEqual([k for k in before if before[k] != after[k]], [key])
                self.assertFalse(report["active_config_changed"])
        self.assertEqual((PROJECT_ROOT / "config/app.json").read_bytes(), runtime)


if __name__ == "__main__":
    unittest.main()
