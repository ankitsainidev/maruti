import json
import zipfile
import time
from tqdm.auto import tqdm

__all__ = ['rep_time', 'read_json', 'write_json', 'unzip']


def rep_time(seconds):
    if seconds >= 3600:
        return time.strftime('%H:%M:%S', time.gmtime(seconds))
    else:
        return time.strftime('%M:%S', time.gmtime(seconds))


def read_json(path):
    '''
    Read Json file as dict.
    '''
    with open(path, 'rb') as file:
        json_dict = json.load(file)
    return json_dict


def write_json(dictionary, path):
    """
    Write dict as a json file
    """
    with open(path, 'w') as fp:
        json.dump(dictionary, fp)


def unzip(zip_path, path='.'):
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc='Extracting '):
            try:
                zf.extract(member, path)
            except zipfile.error as e:
                pass
