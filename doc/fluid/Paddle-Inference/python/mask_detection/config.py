# -*- coding: UTF-8 -*-
import os
# current dir 
PREDICT_FILE_PATH = os.path.split(os.path.realpath(__file__))[0]

# face detect model file dir
DETECT_MODEL_FILE = os.path.join(PREDICT_FILE_PATH,
                                 "models/pyramidbox_lite/model")
# face detect params file dir
DETECT_MODEL_PARAM = os.path.join(PREDICT_FILE_PATH,
                                  "models/pyramidbox_lite/params")
# face mask classify model file dir
MASK_MODEL_FILE = os.path.join(PREDICT_FILE_PATH, "models/mask_detector/model")
# face mask classify params file dir
MASK_MODEL_PARAM = os.path.join(PREDICT_FILE_PATH,
                                "models/mask_detector/params")

# face detect threadhold 
# The face detect model's output is like [a, x1, x2, y1, y2].
# Among them, a represents the confidence of the face. If a > FACE_THRES, means that the area corresponding to the output is a face
FACE_THREAS = 0.6

# Face mask classification threshold
# If the classification result is greater than this threshold, it means that the face is wearing a mask
MASK_THREAS = 0.6

# Before the face detect infernece, the input will be resized to a certain size based on DETECT_INPUT_SHRINK 
DETECT_INPUT_SHRINK = 0.3

FACE_BOX_LINE_WIDTH = 8
TIME_TEXT_SIZE = 50
