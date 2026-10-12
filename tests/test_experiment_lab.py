import unittest

from PIL import Image, ImageDraw

from app.experiment_lab import edge_iou


class EdgeIoUTests(unittest.TestCase):
    def test_identical_images_score_one(self):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 100, 100), outline="black", width=4)
        self.assertEqual(edge_iou(image, image), 1.0)

    def test_score_is_bounded(self):
        first = Image.new("RGB", (128, 128), "white")
        second = Image.new("RGB", (128, 128), "black")
        score = edge_iou(first, second)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
