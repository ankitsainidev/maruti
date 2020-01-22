import json
import zipfile
from tqdm.auto import tqdm


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


def unzip(zipfile, path='.'):
    with zipfile.ZipFile(zipfile) as zf:
        for member in tqdm(zf.infolist(), desc='Extracting '):
            try:
                zf.extract(member, path)
            except zipfile.error as e:
                pass
