import os
import subprocess
from pathlib import Path
import zipfile


def set_variables(credentials: 'lists[str,str]=[username, token]'):
    os.environ['KAGGLE_USERNAME'] = credentials[0]
    os.environ['KAGGLE_KEY'] = credentials[1]


def update_dataset(path, slug, message='new version', clean=False):
    folder = os.path.basename(path)
    path = os.path.dirname(path)
    path = Path(path)
    os.mkdir(path / folder / folder)
    subprocess.call(['kaggle', 'datasets', 'download', '-p',
                     str(path / folder / folder), 'ankitsainiankit/' + slug])

    subprocess.call(['kaggle', 'datasets', 'metadata', '-p',
                     str(path / folder), 'ankitsainiankit/' + slug])

    zipf = os.listdir(path / folder / folder)[0]
    with zipfile.ZipFile(path / folder / folder / zipf, 'r') as zip_ref:
        zip_ref.extractall(path / folder / folder)
    if not clean:
        with zipfile.ZipFile(path / folder / folder / zipf, 'r') as zip_ref:
            zip_ref.extractall(path / folder)

    subprocess.call(['kaggle', 'datasets', 'version',
                     '-m', message, '-p', path / folder])
