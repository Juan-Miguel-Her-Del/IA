"""Control A/B sin inferencia, entrenamiento ni acceso a fotos test."""
import unittest

from training.validation import validate_selection


class ValidationSelectionTests(unittest.TestCase):
    def fixture(self, experiment):
        size = {"A": 421, "B": 884}[experiment]
        checkpoint = {"config": {"experiment": experiment}, "manifest_sha256": "fixture"}
        summary = {"experiment": experiment, "manifest_sha256": "fixture"}
        return checkpoint, summary, [{"split": "val"}] * size

    def test_A_and_B_require_their_own_validation_counts(self):
        for experiment, size in (("A", 421), ("B", 884)):
            checkpoint, summary, rows = self.fixture(experiment)
            self.assertEqual(validate_selection(checkpoint, summary, rows, experiment), (experiment, size))
            with self.assertRaisesRegex(ValueError, "exactamente"):
                validate_selection(checkpoint, summary, rows[:-1], experiment)

    def test_mixed_experiments_are_rejected(self):
        checkpoint, summary, rows = self.fixture("B")
        with self.assertRaisesRegex(ValueError, "no coinciden"):
            validate_selection(checkpoint, summary, rows, "A")
        checkpoint["config"]["experiment"] = "A"
        with self.assertRaisesRegex(ValueError, "no coinciden"):
            validate_selection(checkpoint, summary, rows, "B")

    def test_different_manifest_is_rejected(self):
        checkpoint, summary, rows = self.fixture("B")
        checkpoint["manifest_sha256"] = "other-fixture"
        with self.assertRaisesRegex(ValueError, "manifiesto distinto"):
            validate_selection(checkpoint, summary, rows, "B")


if __name__ == "__main__":
    unittest.main()
