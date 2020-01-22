import maruti
import unittest
import tempfile
from maruti import utils
import os


class UtilsTests(unittest.TestCase):

    def test_open_json(self):
        with tempfile.TemporaryDirectory() as dir:
            # creating dictionary
            sample = {'h': 3, 'd': {'j': 4}}
            path = os.path.join(dir, 'test.json')

            # writing to file
            utils.write_json(sample, path)

            # reading same file
            sample_read = utils.read_json(path)
            self.assertEqual(sample, sample_read)
