import pathlib
import glob
from warnings import warn
from random import choices
import subprocess
import zipfile
import concurrent.futures
import os
from os.path import join
import shlex
import time
from collections import defaultdict
from .utils import unzip, read_json
from .sizes import file_size
from tqdm.auto import tqdm

DATA_PATH = join(os.path.dirname(__file__),'data/')

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
            self.metadata = read_json(metadata_path)
        except FileNotFoundError:
            del metadata_path
            print('metadata file not found.\n Some functionalities may not work.')

        if hasattr(self, 'metadata'):
            self.video_groups = split_videos(self.metadata)

    @staticmethod
    def download_part(part='00', download_path='.', cookies_path=join(DATA_PATH,'kaggle','cookies.txt')):
        dataset_path = f'https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_{part}.zip'
        folder = f'dfdc_train_part_{int(part)}'
        command = f'wget -c --load-cookies {cookies_path} {dataset_path} -P {download_path}'
        command_args = shlex.split(command)
        fp = open(os.devnull, 'w')
        download = subprocess.Popen(command_args, stdout=fp, stderr=fp)
        bar = tqdm(total=10240, desc='Downloading ')
        zip_size = 0
        while download.poll() is None:
            time.sleep(0.1)
            try:
                new_size = int(
                    file_size(download_path+f'/dfdc_train_part_{part}.zip'))
                bar.update(new_size - zip_size)
                zip_size = new_size
            except FileNotFoundError:
                continue
        if download.poll() != 0:
            print('some error')
            print('download', download.poll())
        download.terminate()
        fp.close()
        bar.close()
        return download_path+f'/dfdc_train_part_{part}.zip'

    @classmethod
    def from_part(cls, part='00',
                  cookies_path=join(DATA_PATH,'kaggle','cookies.txt'),
                  download_path='.'):
        folder = f'dfdc_train_part_{int(part)}'

        if os.path.exists(pathlib.Path(download_path)/folder):
            return cls(pathlib.Path(download_path)/folder)
        downloaded_zip = cls.download_part(
            part=part, download_path=download_path, cookies_path=cookies_path)
        unzip(downloaded_zip)
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
