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

import gzip
import argparse
import os

import paddle.v2 as paddle
from mobilenet_without_bn import mobile_net


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='we pre-generate model parameters without BN')
    parser.add_argument(
        '--model_name', help='name the pre-generate model name', type=str)
    return parser.parse_args()


def generate_model(net, model_path):
    with gzip.open(model_path, 'w') as f:
        paddle.parameters.create(net).to_tar(f)
    print 'SUCCESS! ', 'we pre-generate our model without bn in ', model_path


if __name__ == '__main__':
    args = parse_args()

    img_size = 3 * 224 * 224
    class_num = 102
    #create the net base on the model without batch normalization
    net = mobile_net(img_size, class_num)

    base_path = os.path.dirname(os.path.realpath(__file__))
    generate_model(net, os.path.join(base_path, 'models', args.model_name))
