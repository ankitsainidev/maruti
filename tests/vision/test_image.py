import unittest
from cv2 import dnn_Net
from maruti.vision import image

class ImageTests(unittest.TestCase):

    def test_create_net(self):
        self.assertIsInstance(image.create_net(),dnn_Net)

