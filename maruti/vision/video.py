import cv2
import numpy as np
from .. import vision as mvis
from facenet_pytorch import MTCNN
import torch
from PIL import Image
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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


def crop_face(img, points, size: "(h,w)" = None):
    size = size[1], size[0]  # cv2 resize needs (w,h)
    face = img[points[1]:points[3],
               points[0]:points[2]]
    if size is not None:
        face = cv2.resize(face, size)
    return face


def bbox_from_det(det_list):
    working_det = np.array([[0, 0,
                             224, 224]])
    bbox = []
    for detection in det_list:
        if detection is None:
            bbox.append(working_det.astype(int) * 2)
        else:
            bbox.append(detection.astype(int) * 2)
            working_det = detection.copy()
    return bbox


def get_face_frames2(path, start, end, jumps=4, margin=30, mtcnn=None, size: "(h,w)" = (224, 224)):
    cap = cv2.VideoCapture(path)
    f_h, f_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    #
    frame_idx = list(range(start, end))
    detect_idx = list(range(start, end, jumps))
    n_h, n_w = f_h // 2, f_w // 2
    if mtcnn is None:
        mtcnn = MTCNN(select_largest=False, device=device,)

    frames = list(get_frames_from_path(path, frame_idx))
    small_faces = [cv2.resize(frame, (n_w, n_h))
                   for i, frame in enumerate(frames) if i in detect_idx]
    det, conf = mtcnn.detect(small_faces)
    full_det_list = [None] * len(frame_idx)
    det_list = list(map(lambda x: x, det))

    for i, box in zip(detect_idx, det_list):
        full_det_list[i] = box
    bbox = bbox_from_det(full_det_list)
    working_pred = np.array([(f_h // 2) - 112, (f_w // 2) - 112,
                             (f_h // 2) + 112, (f_h // 2) + 112])
    faces = []
    for frame, box in zip(frames, bbox):
        best_pred = box[0]
        best_pred[[0, 1]] -= margin // 2
        best_pred[[2, 3]] += (margin + 1) // 2
        try:
            cropped_faces = crop_face(frame, best_pred, size=size)
            working_pred = best_pred
        except:
            cropped_faces = crop_face(frame, working_pred, size=size)
        faces.append(cropped_faces)

    return faces


def get_face_frames(path, frame_idx, margin=30, mtcnn=None, size: "(h,w)" = (224, 224),):
    """
    Consumes more RAM as it stores all the frames in full resolution.
    Try to detect in small batches if needed.
    """
    # for height and width
    cap = cv2.VideoCapture(path)
    f_h, f_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    #

    n_h, n_w = f_h // 2, f_w // 2
    if mtcnn is None:
        mtcnn = MTCNN(select_largest=False, device=device,)

    frames = list(get_frames_from_path(path, frame_idx))
    small_faces = [cv2.resize(frame, (n_w, n_h)) for frame in frames]
    det, conf = mtcnn.detect(small_faces)
    det_list = list(map(lambda x: x, det))
    bbox = bbox_from_det(det_list)
    working_pred = np.array([(f_h // 2) - 112, (f_w // 2) - 112,
                             (f_h // 2) + 112, (f_h // 2) + 112])
    faces = []
    for frame, box in zip(frames, bbox):
        best_pred = box[0]
        best_pred[[0, 1]] -= margin // 2
        best_pred[[2, 3]] += (margin + 1) // 2
        try:
            cropped_faces = crop_face(frame, best_pred, size=size)
            working_pred = best_pred
        except:
            cropped_faces = crop_face(frame, working_pred, size=size)
        faces.append(cropped_faces)

    return faces
