import numpy as np
import cv2
from functools import lru_cache
from functools import partial
from os.path import join
import os
from PIL import Image
import torch


__all__ = ['brightness_score', 'adjust_brightness',
           'crop_around_point', 'make_grid']

DATA_PATH = join(os.path.dirname(__file__), 'data')


def brightness_score(img):
    '''
    @params:
    img - an array with shape (w/h, w/h, 3)
    '''
    cols, rows = img.shape[:2]
    return np.sum(img) / (255 * cols * rows)


def adjust_brightness(img, min_brightness):
    '''
    Increase of decrease brightness
    @params:
    img - an array with shape (w,h,3)
    '''
    brightness = brightness_score(img)
    ratio = brightness / min_brightness
    return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)


def crop_around_point(img, point, size):
    '''
    crop a rectangle with size centered at point
    @params: size (h,w)
    @params: point (x,y)
    '''
    h, w = img.shape[:2]
    n_h, n_w = size
    r_h, r_w = h, w

    if h < n_h:
        r_h = n_h
    if w < n_w:
        r_w = n_w

    h_ratio = r_h / h
    w_ratio = r_w / w
    if h_ratio > w_ratio:
        r_w = int(r_w * h_ratio / w_ratio)
    elif w_ratio > h_ratio:
        r_h = int(r_h * w_ratio / h_ratio)

    pre_w, post_w = n_w // 2, n_w - (n_w // 2)
    pre_h, post_h = n_h // 2, n_h - (n_h // 2)
    img = cv2.resize(img, (r_w, r_h))
    midx, midy = point
    startX, startY, endX, endY = 0, 0, 0, 0
    if midx - pre_w < 0:
        startX, endX = 0, n_w
    elif midx + post_w - 1 >= r_w:
        startX, endX = r_w - n_w, r_w
    else:
        startX, endX = midx - pre_w, midx + post_w

    if midy - pre_h < 0:
        startY, endY = 0, n_h
    elif midy + post_h - 1 >= r_h:
        startY, endY = r_h - n_h, r_h
    else:
        startY, endY = midy - pre_h, midy + post_h

    return img[startY:endY, startX:endX]


def _unNormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mt = torch.FloatTensor(mean).view(1, 1, 3)
    st = torch.FloatTensor(std).view(1, 1, 3)
    return (((img * st) + mt) * 255).int().numpy().astype(np.uint8)


def make_grid(imgs: '(n,h,w,c) tensor or list of (h,w,c) tensor', cols=8):
    "return numpy array of size (h,w,c) easy for plotting"
    count = len(imgs)
    rows = (count + cols - 1) // cols
    if not (imgs[0] > 5).any():
        imgs = [_unNormalize(img) for img in imgs]
    h, w = imgs[0].shape[:-1]
    new_img_w = h * cols
    new_img_h = w * rows
    new_img = Image.new('RGB', (new_img_w, new_img_h))

    for i in range(len(imgs)):
        img = Image.fromarray(np.array(imgs[i]).astype(np.uint8))
        x = h * (i % cols)
        y = h * (i // cols)
        new_img.paste(img, (x, y))
    return np.array(new_img)
