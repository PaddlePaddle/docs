# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import gzip
import cv2
import numpy as np
import time

import paddle.v2 as paddle
from paddle.v2.inference import Inference

from mobilenet_without_bn import mobile_net as mb_without_bn
from mobilenet_with_bn import mobile_net as mb_with_bn


def infer(net, model):
    with gzip.open(model, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    inference = Inference(output_layer=net, parameters=parameters)

    test_img = load_image(cur_dir + '/image/cat.jpg')
    test_data = []

    test_data.append((test_img, ))
    sum = 0.0
    loops_num = 1
    for i in xrange(loops_num):
        start = time.time()
        probs = inference.infer(field='value', input=test_data)
        end = time.time()
        sum += (end - start)

    print 'time :', sum / loops_num
    print 'class : ', probs[0].argmax()
    print 'prob : ', probs[0].max()


def load_image(file, resize_size=256, crop_size=224, mean_file=None):
    # load image
    im = cv2.imread(file)
    # resize
    h, w = im.shape[:2]
    h_new, w_new = resize_size, resize_size
    if h > w:
        h_new = resize_size * h / w
    else:
        w_new = resize_size * w / h
    im = cv2.resize(im, (h_new, w_new), interpolation=cv2.INTER_CUBIC)
    # crop
    h, w = im.shape[:2]
    h_start = (h - crop_size) / 2
    w_start = (w - crop_size) / 2
    h_end, w_end = h_start + crop_size, w_start + crop_size
    im = im[h_start:h_end, w_start:w_end, :]
    # transpose to CHW order
    mean = np.array([103.94, 116.78, 123.68])
    im = im - mean
    im = im.transpose((2, 0, 1))

    #im = im * 0.017
    return im


if __name__ == '__main__':

    img_size = 3 * 224 * 224
    class_num = 102

    paddle.init(use_gpu=False, trainer_count=1)
    base_path = os.path.dirname(os.path.realpath(__file__))

    #net = mb_without_bn(img_size, class_num)
    #model = os.path.join(base_path, 'models', 'merged_model.tar.gz')

    net = mb_with_bn(img_size, class_num)
    model = os.path.join(base_path, 'models', 'mobilenet_flowers102.tar.gz')

    infer(net, model)
