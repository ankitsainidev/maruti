import numpy as np
import cv2
from functools import lru_cache
from functools import partial
from os.path import join

def brightness_score(img):
    '''
    @params:
    img - an array with shape (w/h, w/h, 3)
    '''
    cols, rows = img.shape[:2]
    return np.sum(img) / (255*cols*rows)


def adjust_brightness(img, min_brightness):
    '''
    Increase of decrease brightness 
    @params:
    img - an array with shape (w,h,3)
    '''
    brightness = brightness_score(img)
    ratio = brightness/min_brightness
    return cv2.convertScaleAbs(img, alpha=1/ratio, beta=0)


@lru_cache(maxsize=2)
def create_net(path='data/cvCafee'):
    '''
    Creates net for face detection.
    '''
    prototxt_path = join(path,'deploy.prototxt.txt')
    model_path = join(path,'res10_300x300_ssd_iter_140000.caffemodel')
    
    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
    return net


def detect_faces(image, net=None):
    '''
    @params: image (h,w,3)
    returns face detection array [., ., condfidence, x1,y1,x2,y2]
    '''

    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net = create_net() if net is None else net
    net.setInput(blob)
    detections = net.forward()
    return detections[0][0]


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
    img = cv2.resize(img, (r_w, r_h))
    midx, midy = point
    startX, startY, endX, endY = 0, 0, 0, 0
    if midx-112 < 0:
        startX, endX = 0, 224
    elif midx+111 >= r_w:
        startX, endX = r_w-224, r_w
    else:
        startX, endX = midx-112, midx+112

    if midy-112 < 0:
        startY, endY = 0, 224
    elif midy+111 >= r_h:
        startY, endY = r_h-224, r_h
    else:
        startY, endY = midy-112, midy+112

    return img[startY:endY, startX:endX]


def detect_sized_face(img, size, brightness_values=[], threshold=0.6, bright_face_crop=True, net=None):
    '''
    '''

    h, w = img.shape[:2]

    detection = detect_faces(img, net)
    i = 0
    best_detection = {
        'confidence': detection[0][2], 'detection': detection, 'image': img}
    while brightness_values and detection[0][2] < threshold:
        if i < len(brightness_values):
            n_image = adjust_brightness(img, brightness_values[i])
            detection = detect_faces(n_image, net)

            if detection[0][2] > best_detection['confidence']:
                best_detection['confidence'] = detection[0][2]
                best_detection['image'] = n_image
                best_detection['detection'] = detection

            i += 1
    if bright_face_crop:
        img = best_detection['image']
        detection = best_detection['detection']
    box = detection[0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    x, y = (startX+endX)/2, (startY + endY)/2
    face = crop_around_point(img, (int(x), int(y)), size)
    return face


def get_face_center(img, brightness_values=[], threshold=0.6, bright_face_crop=True, net=None):
    """ returns(x,y),brightness_value to feed in adjust_brightness"""
    h, w = img.shape[:2]

    detection = detect_faces(img, net)
    i = 0
    best_detection = {
        'confidence': detection[0][2], 'detection': detection, 'image': img, 'brightness_value': brightness_score(img)}
    while brightness_values and detection[0][2] < threshold:
        if i < len(brightness_values):
            n_image = adjust_brightness(img, brightness_values[i])
            detection = detect_faces(n_image, net)

            if detection[0][2] > best_detection['confidence']:
                best_detection['brightness_value'] = brightness_values[i]
                best_detection['confidence'] = detection[0][2]
                best_detection['image'] = n_image
                best_detection['detection'] = detection

            i += 1
    if bright_face_crop:
        img = best_detection['image']
        detection = best_detection['detection']
    box = detection[0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    x, y = (startX+endX)/2, (startY + endY)/2
    return (x, y), detection['brightness_value']