from sys import getsizeof
import os

__all__ = ['dir_size','file_size','var_size']

def byte_to_mb(size):
    return size/(1024**2)


def dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return byte_to_mb(total_size)


def file_size(path):
    file_stats = os.stat(path)
    return byte_to_mb(file_stats.st_size)


def var_size(var):
    return byte_to_mb(getsizeof(var))

__all__ = ['var_size','file_size','dir_size']