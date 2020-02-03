import cv2
import os


class Video(cv2.VideoCapture):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def get_frames(cap: 'cv2.VideoCapture object', frames: 'iterable<int>'):
    """Frame numbers out of the scope will be ignored"""
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = set(frames)
    last_frame = max(frames)
    if frame_count == -1:
        print('The video is corrupt. Closing')
    for i in range(0, frame_count):
        _ = cap.grab()
        if i in frames:
            _, frame = cap.retrieve()
            yield frame
            if i == last_frame:
                break
