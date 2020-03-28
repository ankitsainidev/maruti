import cv2
import numpy as np
from .. import vision as mvis
from facenet_pytorch import MTCNN
import torch
from PIL import Image
from collections import defaultdict
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Video(cv2.VideoCapture):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def vid_info(path):
    "return frame_count, (h, w)"
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    return frame_count, (h, w)


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
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    if size:
        size = size[1], size[0]  # cv2 resize needs (w,h)
    face = img[points[1]:points[3],
               points[0]:points[2]]
    if size is not None:
        face = cv2.resize(face, size,)
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


def _face_from_det(frame_idx, detect_idx, frames, det_list,  f_h, f_w, margin=30, size=(224, 224), mtcnn=None):
    start = frame_idx[0]
    n_h, n_w = f_h // 2, f_w // 2
    full_det_list = [None] * len(frame_idx)

    # first frame should be correct so it can compunsate upcomings
    if det_list[0] is None:
        _detection = mtcnn.detect(frames[0])[0]
        if _detection is not None:
            det_list[0] = _detection / 2
    #

    for i, box in zip(detect_idx, det_list):
        full_det_list[i - start] = box
    bbox = bbox_from_det(full_det_list)
    working_pred = np.array([(f_h // 2) - 112, (f_w // 2) - 112,
                             (f_h // 2) + 112, (f_w // 2) + 112])
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


def non_overlapping_ranges(rngs):
    all_idx = set()
    for rng in rngs:
        for i in range(rng[0], rng[1]):
            all_idx.add(i)
    min_i = min(all_idx)
    max_i = max(all_idx)
    non_overlapping_rngs = []
    last = min_i
    start = min_i
    i = min_i + 1
    while i < max_i + 1:
        if i in all_idx:
            last = i
            i += 1
            continue
        else:
            non_overlapping_rngs.append([start, last + 1])
            while i not in all_idx:
                i += 1
            start = i
            last = i
    non_overlapping_rngs.append([start, last + 1])
    return non_overlapping_rngs


def get_face_frames2(path, frame_rngs, jumps=4, margin=30, mtcnn=None, size: "(h,w)" = (224, 224)):
    # for height and width
    cap = cv2.VideoCapture(path)
    f_h, f_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_h, n_w = f_h // 2, f_w // 2
    cap.release()
    #
    non_overlapping_rngs = non_overlapping_ranges(frame_rngs)
    idx2face = defaultdict(lambda: None)
    idx2frame = defaultdict(lambda: None)
    if mtcnn is None:
        mtcnn = MTCNN(select_largest=False, device=device,)

    # getting video frames in one shot
    all_frames_idx = []
    for rng in non_overlapping_rngs:
        all_frames_idx.extend(range(rng[0], rng[1]))
    vid_frames = list(get_frames_from_path(path, all_frames_idx))
    for i, frame in zip(all_frames_idx, vid_frames):
        idx2frame[i] = frame

    # getting detection in one shot
    all_detect_idx = []
    for frame_rng in non_overlapping_rngs:
        all_detect_idx.extend(range(frame_rng[0], frame_rng[1], jumps))
    all_detect_small_frames = [cv2.resize(frame, (n_w, n_h)) for i, frame in zip(
        all_frames_idx, vid_frames) if i in all_detect_idx]
    det, conf = mtcnn.detect(all_detect_small_frames)
    idx2det = defaultdict(lambda: None)
    for i, det in zip(all_detect_idx, det):
        idx2det[i] = det

    # face crop for each non-overlapping range
    for frame_rng in non_overlapping_rngs:
        start, end = frame_rng
        frame_idx = list(range(start, end))
        detect_idx = list(range(start, end, jumps))
        frames = [idx2frame[i] for i in frame_idx]
        det_list = [idx2det[i] for i in detect_idx]
        faces = _face_from_det(
            frame_idx, detect_idx, frames, det_list, f_h, f_w, margin=margin, size=size, mtcnn=mtcnn)
        for i, face in zip(frame_idx, faces):
            idx2face[i] = face

    # distribution to each range
    rng_faces = []
    for rng in frame_rngs:
        curr_rng_faces = []
        for i in range(rng[0], rng[1]):
            curr_rng_faces.append(idx2face[i])
        rng_faces.append(curr_rng_faces)
    return rng_faces


def crop(frame, bb):
    return frame[bb[1]:bb[3], bb[0]:bb[2]]


def toint(bb):
    return [int(i) for i in bb]


def apply_margin(bb, margin, size):
    bb = [max(0,bb[0]-margin),max(0, bb[1] - margin),min(size[0] -1, bb[2]+margin),min(size[1]-1, bb[3]+margin)]
    return bb


def expand_detection(detections, idx, length):
    assert (len(detections) == len(
        idx)), f'length of detection ({len(detections)}) and indices ({len(idx)}) must be same'

    j = 0
    last = detections[j] if detections[j] is not None else []
    final_detections = []

    for i in range(length):
        if i in idx:
            last = detections[idx.index(i)]
            if last is None:
                last = []
        final_detections.append(last)
    return final_detections


def get_all_faces(path: 'str', detections=32, mtcnn=None, margin=20):
    if mtcnn is None:
        mtcnn = MTCNN(select_largest=False, device=device,)

    cap = cv2.VideoCapture(path)
    frames = []
    next_frame = True
    while next_frame:
        next_frame, fr = cap.read()
        if next_frame:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    np_det_idx = np.linspace(0, len(frames), detections,
                             endpoint=False, dtype=int)
    detection_idx = list(map(int, np_det_idx))
    detection_frames = [frame for i, frame in enumerate(
        frames) if i in detection_idx]
    detection = mtcnn.detect(detection_frames)

    detection = detection[0]
    del detection_frames
    detection = expand_detection(detection, detection_idx, len(frames))

    faces = []

    for i, bboxes in enumerate(detection):
        faces.append([])
        for bbox in bboxes:
            bbox = apply_margin(bbox, margin, frames[0].shape[:2])
            faces[-1].append(crop(frames[i], toint(bbox)))

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
                             (f_h // 2) + 112, (f_w // 2) + 112])
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


def get_faces_frames(path, frame_idx, margin=30, mtcnn=None, size: "(h,w)" = (224, 224),):
    """
    Consumes more RAM as it stores all the frames in full resolution.
    Try to detect in small batches if needed.
    """
    # for height and width
    _, (f_h, f_w) = vid_info(path)
    #

    n_h, n_w = f_h // 2, f_w // 2
    if mtcnn is None:
        mtcnn = MTCNN(select_largest=False, device=device,)

    frames = list(get_frames_from_path(path, frame_idx))
    small_faces = [cv2.resize(frame, (n_w, n_h)) for frame in frames]
    det, conf = mtcnn.detect(small_faces)
    det_list = list(map(lambda x: x, det))
    if det_list[0] is None:
        _detection = mtcnn.detect(frames[0])[0]
        if _detection is not None:
            det_list[0] = _detection / 2
    bbox = bbox_from_det(det_list)
    working_pred = np.array([(f_h // 2) - 112, (f_w // 2) - 112,
                             (f_h // 2) + 112, (f_w // 2) + 112])
    faces = []
    for frame, box in zip(frames, bbox):
        all_faces = []
        for face_det in box:
            best_pred = face_det
            best_pred[[0, 1]] -= margin // 2
            best_pred[[2, 3]] += (margin + 1) // 2
            try:
                cropped_faces = crop_face(frame, best_pred, size=size)
                working_pred = best_pred
            except:
                cropped_faces = crop_face(frame, working_pred, size=size)
            all_faces.append(cropped_faces)
        faces.append(all_faces)

    return faces


def _face_from_det(frame_idx, detect_idx, frames, det_list,  f_h, f_w, margin=30, size=(224, 224), mtcnn=None):
    start = frame_idx[0]
    n_h, n_w = f_h // 2, f_w // 2
    full_det_list = [None] * len(frame_idx)

    # first frame should be correct so it can compunsate upcomings
    if det_list[0] is None:
        _detection = mtcnn.detect(frames[0])[0]
        if _detection is not None:
            det_list[0] = _detection / 2
    #

    for i, box in zip(detect_idx, det_list):
        full_det_list[i - start] = box
    bbox = bbox_from_det(full_det_list)
    working_pred = np.array([(f_h // 2) - 112, (f_w // 2) - 112,
                             (f_h // 2) + 112, (f_w // 2) + 112])
    faces = []
    for frame, box in zip(frames, bbox):
        all_faces = []
        for face_det in box:
            best_pred = face_det
            best_pred[[0, 1]] -= margin // 2
            best_pred[[2, 3]] += (margin + 1) // 2
            try:
                cropped_faces = crop_face(frame, best_pred, size=size)
                working_pred = best_pred
            except:
                cropped_faces = crop_face(frame, working_pred, size=size)
            all_faces.append(cropped_faces)
        faces.append(all_faces)

    return faces


def get_faces_frames2(path, frame_rngs, jumps=4, margin=30, mtcnn=None, size: "(h,w)" = (224, 224)):
    # for height and width
    cap = cv2.VideoCapture(path)
    f_h, f_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_h, n_w = f_h // 2, f_w // 2
    cap.release()
    #
    non_overlapping_rngs = non_overlapping_ranges(frame_rngs)
    idx2face = defaultdict(lambda: None)
    idx2frame = defaultdict(lambda: None)
    if mtcnn is None:
        mtcnn = MTCNN(select_largest=False, device=device,)

    # getting video frames in one shot
    all_frames_idx = []
    for rng in non_overlapping_rngs:
        all_frames_idx.extend(range(rng[0], rng[1]))
    vid_frames = list(get_frames_from_path(path, all_frames_idx))
    for i, frame in zip(all_frames_idx, vid_frames):
        idx2frame[i] = frame

    # getting detection in one shot
    all_detect_idx = []
    for frame_rng in non_overlapping_rngs:
        all_detect_idx.extend(range(frame_rng[0], frame_rng[1], jumps))
    all_detect_small_frames = [cv2.resize(frame, (n_w, n_h)) for i, frame in zip(
        all_frames_idx, vid_frames) if i in all_detect_idx]
    det, conf = mtcnn.detect(all_detect_small_frames)
    idx2det = defaultdict(lambda: None)
    for i, det in zip(all_detect_idx, det):
        idx2det[i] = det

    # face crop for each non-overlapping range
    for frame_rng in non_overlapping_rngs:
        start, end = frame_rng
        frame_idx = list(range(start, end))
        detect_idx = list(range(start, end, jumps))
        frames = [idx2frame[i] for i in frame_idx]
        det_list = [idx2det[i] for i in detect_idx]
        faces = _faces_from_det(
            frame_idx, detect_idx, frames, det_list, f_h, f_w, margin=margin, size=size, mtcnn=mtcnn)
        for i, face in zip(frame_idx, faces):
            idx2face[i] = face

    # distribution to each range
    rng_faces = []
    for rng in frame_rngs:
        curr_rng_faces = []
        for i in range(rng[0], rng[1]):
            curr_rng_faces.append(idx2face[i])
        rng_faces.append(curr_rng_faces)
    return rng_faces
