import tempfile
import unittest
from pathlib import Path

from app.engineering_lab import (
    evaluate_design,
    parse_component_specs,
    render_engineering_report,
    save_engineering_report,
    split_features,
)


class EngineeringLabTests(unittest.TestCase):
    def basic_check(self, **overrides):
        values = {
            "enclosure_length": 100,
            "enclosure_width": 45,
            "enclosure_height": 15,
            "clearance": 1.5,
            "component_specs": "Display, 48, 28, 3\nPCB, 55, 30, 1.6\nOversized, 120, 20, 5",
            "required_features": "USB-C port\nDisplay\nTwo buttons",
            "observed_features": "display, USB-C port",
            "human_rating": 4,
            "description": "A small handheld music player for students with physical controls and offline playback.",
            "priorities": ["Compact", "Portable"],
            "envelope_notes": "Approximately 100 x 45 x 15 mm with USB-C on the bottom edge.",
            "avoid_notes": "Avoid a touchscreen and permanently glued enclosure.",
        }
        values.update(overrides)
        return evaluate_design(**values)

    def test_rotated_component_fits_individually(self):
        result = self.basic_check(component_specs="Connector, 3, 7, 20")
        first = result["component_results"][0]
        self.assertTrue(first["fits_individually"])
        self.assertEqual(sorted(first["oriented_dimensions_mm"]), [3.0, 7.0, 20.0])

    def test_oversized_component_is_reported(self):
        result = self.basic_check()
        self.assertFalse(result["component_results"][2]["fits_individually"])
        self.assertEqual(result["component_fit_pct"], 67)

    def test_feature_coverage_is_case_insensitive(self):
        result = self.basic_check()
        self.assertEqual(result["feature_coverage_pct"], 67)
        self.assertEqual(result["missing_features"], ["Two buttons"])

    def test_empty_required_features_are_not_scored(self):
        result = self.basic_check(required_features="", observed_features="")
        self.assertIsNone(result["feature_coverage_pct"])
        self.assertGreaterEqual(result["review_index_0_to_100"], 0)
        self.assertLessEqual(result["review_index_0_to_100"], 100)

    def test_clearance_cannot_consume_whole_enclosure(self):
        with self.assertRaisesRegex(ValueError, "no positive internal space"):
            self.basic_check(clearance=25)

    def test_component_row_requires_four_fields(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            parse_component_specs("Display, 50, 30")

    def test_duplicate_features_are_removed(self):
        self.assertEqual(split_features("USB-C, display\nUSB-C; Display"), ["USB-C", "display"])

    def test_report_can_be_saved_as_bundle(self):
        result = self.basic_check()
        self.assertIn("Limitations and assumptions", render_engineering_report(result))
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(save_engineering_report(result, output_root=tmp))
            self.assertTrue(bundle_path.is_file())
            self.assertGreater(bundle_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
