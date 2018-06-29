import numpy as np
import gzip
import copy
import cv2, os

import paddle.v2 as paddle
from mobilenet_ssd_pascal import net_conf
from config.pascal_voc_conf import cfg

label_lists = open('./config/label_list').readlines()


def _infer(inferer, infer_data, threshold):
    ret = []
    infer_res = inferer.infer(input=infer_data)
    keep_inds = np.where(infer_res[:, 2] >= threshold)[0]
    for idx in keep_inds:
        ret.append([
            infer_res[idx][0], infer_res[idx][1] - 1, infer_res[idx][2],
            infer_res[idx][3], infer_res[idx][4], infer_res[idx][5],
            infer_res[idx][6]
        ])
    return ret


def draw_result(frame, ret_res, h, w):
    print ret_res
    for det_res in ret_res:
        img_idx = int(det_res[0])
        label = int(det_res[1])
        conf_score = det_res[2]
        xmin = int(round(det_res[3] * w))
        ymin = int(round(det_res[4] * h))
        xmax = int(round(det_res[5] * w))
        ymax = int(round(det_res[6] * h))
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                      (0, (1 - xmin) * 255, xmin * 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label_lists[label + 1].strip(),
                    (xmin + 10, ymin + 10), font, 1.0, (255, 0, 0), 2)


def pre_process(img):
    img = cv2.resize(
        img, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)
    # image should be RGB format
    img = img[:, :, ::-1]
    # image shoud be in CHW format
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 1, 0)
    img = img.astype('float32')

    img_mean = np.array(
        [104, 117, 124])[:, np.newaxis, np.newaxis].astype('float32')
    img -= img_mean
    img = img.flatten()
    return img


def infer(model_path, threshold):

    net = net_conf(mode='infer')

    assert os.path.isfile(model_path), 'Invalid model.'
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))

    #build the inference network
    inferer = paddle.inference.Inference(
        output_layer=net, parameters=parameters)

    test_data = []

    frame = cv2.imread('./images/example.jpg')

    h, w, _ = frame.shape
    img = copy.deepcopy(frame)

    # preprocess the image
    img = pre_process(img)
    test_data.append([img])

    #the forward process
    ret_res = _infer(inferer, test_data, threshold)

    draw_result(frame, ret_res, h, w)
    cv2.imwrite('./images/result.jpg', frame)


if __name__ == "__main__":
    # init paddle environment
    paddle.init(use_gpu=False, trainer_count=1, gpu_id=3)

    infer(model_path='./mobilenet_ssd_pascal.tar.gz', threshold=0.3)
