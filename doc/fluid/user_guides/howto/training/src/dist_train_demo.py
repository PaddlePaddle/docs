#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.

from __future__ import print_function

import paddle.fluid.core as core
import math
import os
import sys

import numpy

import paddle
import paddle.fluid as fluid

BATCH_SIZE = 64
PASS_NUM = 1

def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc

def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train(use_cuda, role, endpoints, current_endpoint, trainer_id, trainers):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    prediction, avg_loss, acc = conv_net(img, label)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    t = fluid.DistributeTranspiler()
    t.transpile(trainer_id, pservers=endpoints, trainers=trainers)
    if role == "pserver":
        prog = t.get_pserver_program(current_endpoint)
        startup = t.get_startup_program(current_endpoint, pserver_program=prog)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup)
        exe.run(prog)
    elif role == "trainer":
        prog = t.get_trainer_program()
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
        feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
        exe.run(fluid.default_startup_program())
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                acc_np, avg_loss_np = exe.run(prog,
                                            feed=feeder.feed(data),
                                            fetch_list=[acc, avg_loss])
                if (batch_id + 1) % 10 == 0:
                    print(
                        'PassID {0:1}, BatchID {1:04}, Loss {2:2.2}, Acc {3:2.2}'.
                        format(pass_id, batch_id + 1,
                                float(avg_loss_np.mean()), float(acc_np.mean())))

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python %s role endpoints current_endpoint trainer_id trainers" % sys.argv[0])
        exit(0)
    role, endpoints, current_endpoint, trainer_id, trainers = \
        sys.argv[1:]
    train(True, role, endpoints, current_endpoint, int(trainer_id), int(trainers))

