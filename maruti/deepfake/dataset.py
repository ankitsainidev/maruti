import pathlib
from warnings import warn
import subprocess

import os
from os.path import join
import torch
import shlex
import time
from collections import defaultdict
from ..vision.video import get_frames_from_path, get_frames
import random
from ..utils import unzip, read_json
from ..sizes import file_size
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

DATA_PATH = join(os.path.dirname(__file__), 'data/')
__all__ = ['split_videos', 'VideoDataset']


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

        metadata_path = metadata_path if metadata_path else self.path / 'metadata.json'
        try:
            self.metadata = read_json(metadata_path)
        except FileNotFoundError:
            del metadata_path
            print('metadata file not found.\n Some functionalities may not work.')

        if hasattr(self, 'metadata'):
            self.video_groups = split_videos(self.metadata)

    @staticmethod
    def download_part(part='00', download_path='.', cookies_path=join(DATA_PATH, 'kaggle', 'cookies.txt')):
        dataset_path = f'https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_{part}.zip'
        # folder = f'dfdc_train_part_{int(part)}'
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
                    file_size(download_path + f'/dfdc_train_part_{part}.zip'))
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
        return download_path + f'/dfdc_train_part_{part}.zip'

    @classmethod
    def from_part(cls, part='00',
                  cookies_path=join(DATA_PATH, 'kaggle', 'cookies.txt'),
                  download_path='.'):
        folder = f'dfdc_train_part_{int(part)}'

        if os.path.exists(pathlib.Path(download_path) / folder):
            return cls(pathlib.Path(download_path) / folder)
        downloaded_zip = cls.download_part(
            part=part, download_path=download_path, cookies_path=cookies_path)
        unzip(downloaded_zip)
        os.remove(download_path + f'/dfdc_train_part_{part}.zip')
        path = pathlib.Path(download_path) / folder
        return cls(path)

    def __len__(self):
        return len(self.video_paths)

    def n_groups(self, n, k=-1):
        '''
        returns random n real-fake pairs by default.
        else starting from k.
        '''
        if k != -1:
            if n + k >= len(self.video_groups):
                warn(RuntimeWarning(
                    'n+k is greater then video length. Returning available'))
                n = len(self.video_groups) - k - 1
            return self.video_groups[k:n + k]
        if n >= len(self.video_groups):
            warn(RuntimeWarning('n is greater then total groups. Returning available'))
            n = len(self.video_groups) - 1
        return choices(self.video_groups, k=n)


class VidFromPathLoader:
    """ Loader to use with DeepfakeDataset class"""

    def __init__(self, paths, img_reader=None):
        """paths as {'00':/part/00,'01'..}"""
        self.path = paths
        self.img_reader = self.img_reader if img_reader is None else img_reader

    @staticmethod
    def img_reader(path, split='val', max_limit=40):
        frame_no = 0 if split == 'val' else random.randint(0, max_limit)
        frame = list(get_frames_from_path(
            path, [frame_no]))[0]
        return frame

    @staticmethod
    def img_group_reader(path, split='val', mode='distributed', num_frames=4, mode_info=[None]):
        """use with partial to set mode
        mode info: distributed -> No Use
                    forward -> {jumps, index:0, readjust_jumps: True}
                    backward -> {jumps, index:-1, readjust_jumps: True} -1 refers to end"""
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if mode == 'distributed':
            frames = np.linspace(0, frame_count - 1, num_frames,  dtype=int)
        elif mode == 'forward':
            start = mode_info.get('index', 0)
            adjust = mode_info.get('readjust_jumps', True)
            jumps = mode_info['jumps']
            if adjust:
                frames = np.linspace(start, min(
                    frame_count - 1, start + (num_frames - 1) * jumps), num_frames, dtype=int)
            else:
                frames = np.linspace(
                    start, start + (num_frames - 1) * jumps, num_frames, dtype=int)
        elif mode == 'backward':
            end = mode_info.get('index', frame_count)
            adjust = mode_info.get('readjust_jumps', True)
            jumps = mode_info['jumps']
            if adjust:
                frames = np.linspace(
                    max(0, start - (num_frames - 1) * jumps), end, num_frames, dtype=int)
            else:
                frames = np.linspace(
                    start + (num_frames - 1) * jumps, num_frames, end, dtype=int)
        return get_frames(cap, frames, 'rgb')

    def __call__(self, metadata, video, split='val'):
        vid_meta = metadata[video]
        video_path = join(self.path[vid_meta['part']], video)

        return self.img_reader(video_path, split)


class DeepfakeDataset(Dataset):
    """Methods 'f12' r1-f1, r1-f2..,(default)
               'f..' r1-f1/f2/f3..
               'f1' r1-f1,
               'ff' r f1 f2 f3..

        Metadata 'split'(train-val),'label'(FAKE-REAL),'fakes'([video,video])
        loader func(metadata,video,split)->input
        error_handler func(self, index, error)->(input, label)"""
    iteration = 0

    def __init__(self, metadata, loader, transform=None, split='train', method='f12', error_handler=None):
        self.transform = transform
        self.split = split
        self.loader = loader
        self.method = method
        self.error_handler = error_handler
        self.metadata = metadata
        self.dataset = []
        real_videos = filter(
            lambda x: metadata[x]['split'] == split, list(split_videos(metadata)))
        for real_video in real_videos:
            fake_videos = list(metadata[real_video]['fakes'])
            self.dataset.append(real_video)
            if method == 'f12':
                self.dataset.append(
                    fake_videos[self.iteration % len(fake_videos)])
            elif method == 'f..':
                self.dataset.append(random.choice(fake_videos))
            elif method == 'f1':
                self.dataset.append(fake_videos[0])
            elif method == 'ff':
                for fake_video in fake_videos:
                    self.dataset.append(fake_video)
            else:
                raise ValueError(
                    'Not a valid method. Choose from f12, f.., f1, ff')

    def __getitem__(self, i):
        if i == 0:
            self.iteration += 1

        try:
            img = self.loader(self.metadata, self.dataset[i], split=self.split)
            label = torch.tensor(
                [float(self.metadata[self.dataset[i]]['label'] == 'FAKE')])
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        except Exception as e:
            if self.error_handler is None:
                def default_error_handler(obj, x, e):
                    print(f'on video {x} error: {e}')
                    return self[random.randint(1, len(self) - 1)]
                self.error_handler = default_error_handler
            return self.error_handler(self, i, e)

    def __len__(self):
        return len(self.dataset)
