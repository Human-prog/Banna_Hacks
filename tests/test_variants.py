"""Offline tests for multi-variant helpers; no hosted image generation needed."""

from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image, ImageDraw

from app.variants import (
    add_variant_direction,
    create_comparison_sheet,
    create_variant_bundle,
    variant_direction,
)


class VariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = []
        for idx, color in enumerate(("#cbd5e1", "#93c5fd", "#a7f3d0"), start=1):
            path = self.root / f"concept_{idx}.png"
            image = Image.new("RGB", (320 + idx * 10, 220), color)
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((50, 50, 270, 170), outline="#111827", width=5)
            draw.text((80, 90), f"Concept {idx}", fill="#111827")
            image.save(path)
            self.images.append(path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_three_directions_are_distinct(self) -> None:
        titles = [variant_direction(idx)[0] for idx in (1, 2, 3)]
        self.assertEqual(len(set(titles)), 3)
        prompt, title = add_variant_direction("base prompt", 2)
        self.assertIn("base prompt", prompt)
        self.assertIn("Alternative form exploration", prompt)
        self.assertEqual(title, "Alternative form exploration")

    def test_invalid_variant_indices_are_rejected(self) -> None:
        for idx in (0, 4, -1, True):
            with self.subTest(index=idx):
                with self.assertRaises(ValueError):
                    variant_direction(idx)

    def test_comparison_sheet_supports_one_to_three_images(self) -> None:
        for count in (1, 2, 3):
            with self.subTest(count=count):
                out = self.root / f"sheet_{count}.png"
                created = create_comparison_sheet(
                    self.images[:count],
                    [f"Variant {i}" for i in range(1, count + 1)],
                    out,
                )
                self.assertTrue(created.is_file())
                with Image.open(created) as image:
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)

    def test_bundle_contains_variant_images_and_comparison_sheet(self) -> None:
        sheet = create_comparison_sheet(self.images[:2], ["One", "Two"], self.root / "sheet.png")
        bundle = create_variant_bundle("Demo device", self.images[:2], ["One", "Two"], sheet, self.root / "bundle.zip")
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
        self.assertIn("variants/01_one.png", names)
        self.assertIn("variants/02_two.png", names)
        self.assertIn("comparison_sheet.png", names)
        self.assertIn("README.md", names)

    def test_label_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching label"):
            create_comparison_sheet(self.images[:2], ["Only one"], self.root / "bad.png")


if __name__ == "__main__":
    unittest.main()
