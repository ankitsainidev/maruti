import unittest
import cv2
from cv2 import dnn_Net
from maruti.vision import image
import os
TEST_DATA_PATH = 'test_data'


class ImageTests(unittest.TestCase):

    def setUp(self):
        self.img_path = os.path.join(TEST_DATA_PATH, 'img1.jpeg')
        self.img = cv2.imread(self.img_path)

    def test_create_net(self):
        self.assertIsInstance(image.create_net(), dnn_Net)

    def test_brightness_score(self):
        self.assertAlmostEqual(
            image.brightness_score(self.img), 1.76, delta=1e-2)

    def test_adjust_brightness(self):
        brightness = image.brightness_score(self.img)
        new_img = image.adjust_brightness(self.img, 2*brightness)
        self.assertGreaterEqual(image.brightness_score(new_img), brightness)

    def test_crop_around_point(self):
        h, w = self.img.shape[:2]
        points = [(0, 0), (h-1, w-1), (h//2, w//2)]
        sizes = [(224, 224), (160, 160), (3000, 4000)]
        for point in points:
            for size in sizes:
                cropped = image.crop_around_point(self.img, point, size)
                self.assertEqual(size, cropped.shape[:2])

    def test_get_face_center(self):
        old_brightness = image.brightness_score(self.img)
        (x, y), brightness = image.get_face_center(self.img)
        self.assertEqual(old_brightness, brightness)

    def test_detect_sized_rescaled_face(self):
        sizes = [(224, 224), (160, 160), (3000, 4000)]
        for size in sizes[::-1]:
            face = image.detect_sized_rescaled_face(self.img, size,rescale_factor=2)
            self.assertEqual(size, face.shape[:2])