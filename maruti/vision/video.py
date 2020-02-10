import cv2
import os
import numpy as np


class Video(cv2.VideoCapture):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def get_frames(cap: 'cv2.VideoCapture object', frames: 'iterable<int>', code='rgb', start_frame=0):
    """Frame numbers out of the scope will be ignored"""
    curr_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if curr_index != start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_index)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = set(frames)
    last_frame = max(frames)
    if frame_count == 0:
        raise Exception('The video is corrupt. Closing')
    for i in range(curr_index, frame_count):
        _ = cap.grab()
        if i in frames:
            _, frame = cap.retrieve()
            if code == 'rgb':
                yield np.ascontiguousarray(frame[..., ::-1])
            else:
                yield frame
            if i == last_frame:
                cap.release()
                break
    cap.release()


def get_frames_from_path(path: 'str or posix', frames: 'iterable<int>', code='rgb'):
    cap = cv2.VideoCapture(str(path))
    return get_frames(cap, frames, code)
