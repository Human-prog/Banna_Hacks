"""Tests the Studio variant orchestrator using mocked model calls."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import app.main as main


class VariantPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_studio_returns_all_nine_ui_outputs_for_two_variants(self) -> None:
        def fake_generate(*args, variant_index=1, **kwargs):
            run_dir = self.root / f"run_{variant_index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            image_path = run_dir / "concept.png"
            image = Image.new("RGB", (300, 200), (200, 220 - variant_index * 10, 240))
            ImageDraw.Draw(image).rectangle((50, 40, 250, 160), outline="black", width=5)
            image.save(image_path)
            brief_path = run_dir / "brief.md"
            brief_path.write_text(f"# Fake variant {variant_index}\n", encoding="utf-8")
            return str(image_path), f"status {variant_index}", f"brief {variant_index}", None, []

        with patch.object(main, "OUTPUT_DIR", self.root / "outputs"), \
             patch.object(main, "generate_concept", side_effect=fake_generate), \
             patch.object(main, "load_history", return_value=[]):
            outputs = main.generate_from_studio(
                None, "Demo device", "A small test product.", [], [], [], [],
                "", "", "", "Studio product render", main.PROMPT_MODE,
                "Clean sketch", 2,
            )

        self.assertEqual(len(outputs), 9)
        first_image, status, brief, bundle, history, quick_preview, short_status, gallery, sheet = outputs
        self.assertIn("2 of 2", status)
        self.assertIn("Variant 1", brief)
        self.assertEqual(history, [])
        self.assertEqual(first_image, quick_preview)
        self.assertEqual(len(gallery), 2)
        self.assertTrue(Path(bundle).is_file())
        self.assertTrue(Path(sheet).is_file())
        self.assertIn("2/2", short_status)


if __name__ == "__main__":
    unittest.main()
