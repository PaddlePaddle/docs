import cv2
import numpy as np
from PIL import Image
import math

FACE_H = 128
FACE_W = 128


def face_detect_preprocess(img, shrink=1.0):
    # BGR  
    img_shape = img.shape
    img = cv2.resize(
        img, (int(img_shape[1] * shrink), int(img_shape[0] * shrink)),
        interpolation=cv2.INTER_CUBIC)

    # HWC -> CHW
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 1, 0)

    # RBG to BGR
    mean = [104., 117., 123.]
    scale = 0.007843
    img = img.astype('float32')
    img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img = img * scale
    img = img[np.newaxis, :]
    return img


index = 0


def mask_classify_preprocess(img, pts):
    # BGR  
    img_face, _ = crop(img, pts)
    t_img_face = img_face.copy()
    #   global index
    #   index += 1
    #   cv2.imwrite(str(index)+ ".jpg", img_face)
    img_face = img_face / 256.
    # HWC -> CHW
    img_face = np.swapaxes(img_face, 1, 2)
    img_face = np.swapaxes(img_face, 1, 0)

    # RBG to BGR
    mean = [0.5, 0.5, 0.5]
    img_face = img_face.astype('float32')
    img_face -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img_face = img_face.reshape(-1, 3, FACE_H, FACE_W)
    return img_face, t_img_face


#def crop(image, pts, shift=0, scale=1.38, rotate=0, res_width=128, res_height=128):
def crop(image,
         pts,
         shift=0,
         scale=1.5,
         rotate=0,
         res_width=FACE_W,
         res_height=FACE_H):
    res = (res_width, res_height)
    idx1 = 0
    idx2 = 1
    # angle
    alpha = 0
    if pts[idx2, 0] != -1 and pts[idx2, 1] != -1 and pts[
            idx1, 0] != -1 and pts[idx1, 1] != -1:
        alpha = math.atan2(pts[idx2, 1] - pts[idx1, 1],
                           pts[idx2, 0] - pts[idx1, 0]) * 180 / math.pi
    pts[pts == -1] = np.inf
    coord_min = np.min(pts, 0)
    pts[pts == np.inf] = -1
    coord_max = np.max(pts, 0)
    # coordinates of center point
    c = np.array([
        coord_max[0] - (coord_max[0] - coord_min[0]) / 2,
        coord_max[1] - (coord_max[1] - coord_min[1]) / 2
    ])  # center
    max_wh = max((coord_max[0] - coord_min[0]) / 2,
                 (coord_max[1] - coord_min[1]) / 2)
    # Shift the center point, rot add eyes angle
    c = c + shift * max_wh
    rotate = rotate + alpha
    M = cv2.getRotationMatrix2D((c[0], c[1]), rotate,
                                res[0] / (2 * max_wh * scale))
    M[0, 2] = M[0, 2] - (c[0] - res[0] / 2.0)
    M[1, 2] = M[1, 2] - (c[1] - res[0] / 2.0)
    image_out = cv2.warpAffine(image, M, res)
    return image_out, M
