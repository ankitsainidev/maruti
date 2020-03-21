import pathlib
from warnings import warn
import subprocess
import maruti
import os
from os.path import join
from PIL import Image
import torch
import shlex
import time
from collections import defaultdict
from ..vision.video import get_frames_from_path, get_frames
import random
from ..utils import unzip, read_json
from ..sizes import file_size
import numpy as np
import cv2
from tqdm.auto import tqdm
from torchvision import transforms as torch_transforms
from torch.utils.data import Dataset
from ..torch.utils import def_norm as normalize
DATA_PATH = join(os.path.dirname(__file__), 'data/')
__all__ = ['split_videos', 'VideoDataset', 'transform', 'group_transform']

transform = {
    'train': torch_transforms.Compose(
        [
            torch_transforms.ToPILImage(),
            torch_transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            torch_transforms.RandomHorizontalFlip(),
            torch_transforms.RandomResizedCrop((224, 224), scale=(0.65, 1.0)),
            torch_transforms.ToTensor(),
            normalize,
        ]
    ),
    'val': torch_transforms.Compose([
        torch_transforms.ToTensor(),
        normalize, ]
    )
}
group_transform = {
    'train': lambda x: torch.stack(list(map(transform['train'], x))),
    'val': lambda x: torch.stack(list(map(transform['val'], x)))
}


class ImageReader:

    def __init__(self, path, metadata, is_path_cache=False, vb=True, ignore_frame_errors=False):
        self.vid2part = {}
        self.meta = metadata
        self.ignore_frame_errors = ignore_frame_errors

        if not is_path_cache:
            parts = os.listdir(path)
            assert len(parts) > 0, 'no files found'
            start = time.perf_counter()

            for part in parts:
                path_to_part = os.path.join(path, part)
                imgs = os.listdir(path_to_part)

                for img in imgs:
                    self.vid2part[self.vid_name(img)] = path_to_part

            end = time.perf_counter()
            if vb:
                print('Total time taken:', (end - start) / 60, 'mins')

        else:
            self.vid2part = maruti.read_json(path)

    def is_real(self, vid):
        return self.meta[vid]['label'] == 'REAL'

    def is_fake(self, vid):
        return not self.is_real(vid)

    def is_error(self, vid):
        return 'error' in self.meta[vid]

    def vid_name(self, img_name):
        name = img_name.split('_')[0]
        return name + '.mp4'

    def create_name(self, vid, frame, person):
        return f'{vid[:-4]}_{frame}_{person}.jpg'

    def total_persons(self, vid):
        if self.is_real(vid):
            return self.meta[vid]['pc']

        orig_vid = self.meta[vid]['original']
        return self.meta[orig_vid]['pc']

    def random_person(self, vid, frame):
        person = random.choice(range(self.total_persons(vid)))
        return self.get_image(vid, frame, person)

    def random_img(self, vid):
        frame = random.choice(range(self.total_frames(vid)))
        person = random.choice(range(self.total_persons(vid)))
        return self.get_image(vid, frame, person)

    def sample(self):
        vid = random.choice(list(self.vid2part))
        while self.is_error(vid):
            vid = random.choice(list(self.vid2part))
        frame = random.choice(range(self.total_frames(vid)))
        person = random.choice(range(self.total_persons(vid)))
        return self.get_image(vid, frame, person)

    def total_frames(self, vid):
        return self.meta[vid]['fc'] - 1

    def create_absolute(self, name):
        path = os.path.join(self.vid2part[self.vid_name(name)], name)
        return path

    def get_image(self, vid, frame, person):
        if self.total_persons(vid) <= person:
            raise Exception('Not Enough Persons')

        if self.total_frames(vid) <= frame:
            if self.ignore_frame_errors:
                frame = self.total_frames(vid) - 1
            else:
                raise Exception('Not Enough Frames')

        img = self.create_name(vid, frame, person)
        path = self.create_absolute(img)
        return Image.open(path)


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
        unzip(downloaded_zip, path=download_path)
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
                    print(f'on video {self.dataset[x]} error: {e}')
                    return self[random.randint(1, len(self) - 1)]
                self.error_handler = default_error_handler
            return self.error_handler(self, i, e)

    def __len__(self):
        return len(self.dataset)
