from .general import open_json
import pathlib
import glob
from warnings import warn
from random import choices
import subprocess
import zipfile
import os
from collections import defaultdict


def split_videos(meta_file):
    '''
    Groups real-fake videos in dictionary
    '''
    split = defaultdict(lambda: set())
    for vid in meta_file:
        if meta_file[vid]['label'] == 'FAKE':
            split[meta_file[vid]['original']].add(vid)
    return split


class VideoDataset:
    '''
    create dataset from videos and metadata.
    @params:

    To download and create use VideoDataset.from_part method
    '''

    def __init__(self, path, metadata_path=None):
        self.path = pathlib.Path(path)
        self.video_paths = list(self.path.glob('*.mp4'))

        metadata_path = metadata_path if metadata_path else self.path/'metadata.json'
        try:
            self.metadata = open_json(metadata_path)
        except FileNotFoundError:
            del metadata_path
            print('metadata file not found.\n Some functionalities may not work.')

        if hasattr(self, 'metadata'):
            self.video_groups = split_videos(self.metadata)

    @classmethod
    def from_part(cls, part='00',
                  cookies_path='data/cookies.txt',
                  download_path='.'):
        dataset_path = f'https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_{part}.zip'
        folder = f'dfdc_train_part_{int(part)}'
        if os.path.exists(pathlib.Path(download_path)/folder):
            return cls(pathlib.Path(download_path)/folder)
        if os.path.exists(f'dfdc_train_part_{part}.zip'):
            os.remove(f'dfdc_train_part_{part}.zip')
        subprocess.call(['wget', '--load-cookies', cookies_path,
                         dataset_path, '-P', download_path])
        with zipfile.ZipFile(download_path+f'/dfdc_train_part_{part}.zip', 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(download_path+f'/dfdc_train_part_{part}.zip')
        path = pathlib.Path(download_path)/folder

        return cls(path)

    def __len__(self):
        return len(self.video_paths)

    def n_groups(self, n, k=-1):
        '''
        returns random n real-fake pairs by default.
        else starting from k.
        '''
        if k != -1:
            if n+k >= len(self.video_groups):
                warn(RuntimeWarning(
                    'n+k is greater then video length. Returning available'))
                n = len(self.video_groups)-k-1
            return self.video_groups[k:n+k]
        if n >= len(self.video_groups):
            warn(RuntimeWarning('n is greater then total groups. Returning available'))
            n = len(self.video_groups)-1
        return choices(self.video_groups, k=n)
