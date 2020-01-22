import unittest
import tempfile
import os
from maruti import sizes


class DeepfakeTest(unittest.TestCase):

    def test_byte_to_mb(self):
        self.assertEqual(sizes.byte_to_mb(1024*1024), 1)
        self.assertAlmostEqual(sizes.byte_to_mb(1024),
                               0.0009765624, delta=1e-8)

    def test_sizes(self):
        with tempfile.TemporaryDirectory() as dir:
            # dir test
            sizes.dir_size(dir)
            sizes.dir_size()

            # file test
            with open(os.path.join(dir, 'test_file.txt'), 'w') as f:
                f.write("It's a test")
            sizes.file_size(os.path.join(dir, 'test_file.txt'))

            # var test
            sizes.var_size(dir)
