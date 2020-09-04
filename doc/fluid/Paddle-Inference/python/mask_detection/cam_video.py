# -*- coding: UTF-8 -*-

import cv2
from mask_detect import MaskPred

# The MaskPred class implements the function of face mask detection,
# including face detection and face mask classification
mp = MaskPred(True, True, 0)
# Turn on the first camera, 0 means device ID
cap = cv2.VideoCapture(0)
cv2.namedWindow('Mask Detect')

while True:
    ret, frame = cap.read()
    if cv2.waitKey(10) == ord("q"):
        break
    result = mp.run(frame)
    cv2.imshow("image", result['img'])
