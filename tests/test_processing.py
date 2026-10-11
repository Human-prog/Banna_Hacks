"""Fast offline tests for VisionForge's sketch preparation pipeline."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from app.processing import MAX_SIDE, PROFILES, prepare_sketch


class ProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.sketch_path = self.root / "sample.png"
        image = Image.new("RGB", (1600, 1200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((250, 250, 1350, 900), outline="black", width=8)
        draw.line((250, 250, 1350, 900), fill="black", width=5)
        draw.ellipse((650, 450, 950, 750), outline="black", width=6)
        image.save(self.sketch_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_all_profiles_return_rgb_image_and_edge_map(self) -> None:
        for profile in PROFILES:
            with self.subTest(profile=profile):
                result = prepare_sketch(self.sketch_path, profile)
                self.assertEqual(result.processed.mode, "RGB")
                self.assertEqual(result.edges.mode, "L")
                self.assertEqual(result.processed.size, result.edges.size)
                self.assertEqual(result.report["profile"], profile)
                self.assertIn("edge_density_pct", result.report)

    def test_longest_edge_is_capped(self) -> None:
        result = prepare_sketch(self.sketch_path, "Clean sketch")
        self.assertLessEqual(max(result.processed.size), MAX_SIDE)
        self.assertEqual(result.report["source_width"], 1600)
        self.assertEqual(result.report["source_height"], 1200)

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "could not be found"):
            prepare_sketch(self.root / "missing.png")

    def test_tiny_image_is_rejected(self) -> None:
        tiny = self.root / "tiny.png"
        Image.new("RGB", (32, 32), "white").save(tiny)
        with self.assertRaisesRegex(ValueError, "too small"):
            prepare_sketch(tiny)

    def test_non_image_is_rejected(self) -> None:
        text_file = self.root / "not-an-image.txt"
        text_file.write_text("not an image", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a supported image"):
            prepare_sketch(text_file)


if __name__ == "__main__":
    unittest.main()
