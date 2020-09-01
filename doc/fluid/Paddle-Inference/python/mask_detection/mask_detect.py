import cv2, os, sys
import numpy as np
from models.pd_model import Model
from models.preprocess import face_detect_preprocess, mask_classify_preprocess
from PIL import Image
from PIL import ImageDraw, ImageFont
import datetime
from config import *


class MaskPred:
    def __init__(self, use_mkldnn=True, use_gpu=False, device_id=0):
        # face detector
        self.face_detector = Model(DETECT_MODEL_FILE, DETECT_MODEL_PARAM,
                                   use_mkldnn, use_gpu, device_id)
        self.face_threas = FACE_THREAS
        # face mask classify 
        self.mask_classify = Model(MASK_MODEL_FILE, MASK_MODEL_PARAM,
                                   use_mkldnn, use_gpu, device_id)
        self.mask_threas = MASK_THREAS
        self.index = 0

    def get_faces(self, data, h, w):
        faces_loc = []
        for d in data:
            if d[1] >= self.face_threas:
                x_min = max(d[2] * w, 0)
                y_min = max(d[3] * h, 0)
                x_h = min((d[4] - d[2]) * w, w)
                y_w = min((d[5] - d[3]) * h, h)
                faces_loc.append([int(x_min), int(y_min), int(x_h), int(y_w)])
        return faces_loc

    def draw_boxes(self, img, boxes):
        h, w, _ = img.shape
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        CUR_FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
        for box in boxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[0] + box[2]
            y_max = box[1] + box[3]
            (left, right, top, bottom) = (x_min, x_max, y_min, y_max)
            color = "red"
            if box[4] < self.mask_threas:
                color = "blue"
            draw.line(
                [(left - 10, top - 10), (left - 10, bottom + 10),
                 (right + 10, bottom + 10), (right + 10, top - 10),
                 (left - 10, top - 10)],
                width=FACE_BOX_LINE_WIDTH,
                fill=color)
            conf_text = str(box[4])

            draw.text(
                [left, top - 50],
                conf_text,
                font=ImageFont.truetype(
                    os.path.join(CUR_FILE_PATH,
                                 "assets/VCR_OSD_MONO_1.001.ttf"),
                    size=30),
                fill="#ff0000")
            cur = datetime.datetime.now()
            cur = str(cur)
            draw.text(
                [10, 10],
                cur,
                font=ImageFont.truetype(
                    os.path.join(CUR_FILE_PATH,
                                 "assets/VCR_OSD_MONO_1.001.ttf"),
                    size=TIME_TEXT_SIZE),
                fill="#ff0000")
        img = np.asarray(image)
        return img

    # do face detect and mask classify
    def run(self, img):
        h, w, c = img.shape
        img_t = face_detect_preprocess(img, DETECT_INPUT_SHRINK)
        results = self.face_detector.run([img_t])
        faces = self.get_faces(results[0], h, w)
        faces_mask_loc_conf = []
        all_with_mask = True
        for loc in faces:
            # (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)
            pts = np.array([
                loc[0], loc[1], loc[2] + loc[0], loc[1], loc[0],
                loc[1] + loc[3], loc[2] + loc[0], loc[1] + loc[3]
            ]).reshape(4, 2).astype(np.float32)
            face_img_t, temp_face = mask_classify_preprocess(img, pts)
            mask_results = self.mask_classify.run([face_img_t])
            mask_conf = mask_results[0]
            temp_loc = loc
            if (mask_conf[0][1] < self.mask_threas):
                all_with_mask = False
            temp_loc.append(mask_conf[0][1])
            faces_mask_loc_conf.append(temp_loc)

        result_dict = {
            "all_with_mask": all_with_mask,
            "loc_conf": faces_mask_loc_conf
        }
        result_dict['face_num'] = len(faces_mask_loc_conf)
        img = self.draw_boxes(img, faces_mask_loc_conf)
        result_dict['img'] = img
        return result_dict


if __name__ == "__main__":
    mp = MaskPred(True, True, 0)
    img = cv2.imread("./assets/test_mask_detection.jpg")
    result = mp.run(img)
    print(result['loc_conf'])

    if not result["all_with_mask"]:
        result_img = result['img']
        h, w, _ = result_img.shape
        result_img = cv2.resize(result_img, (int(w * 0.6), int(h * 0.6)))
        cv2.imshow("image", result_img)
        cv2.waitKey(0)
