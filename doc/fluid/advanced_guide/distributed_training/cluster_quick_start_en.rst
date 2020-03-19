Quick start for distributed training
====================================

Distributed training with Fleet API
-----------------------------------

Since Paddle Fluid `Release
1.5.1 <https://github.com/PaddlePaddle/Paddle/releases/tag/v1.5.1>`__,
it is officially recommended to use the Fleet API for distributed
training. For the introduction of the Fleet API, please refer to `Fleet
Design Doc <https://github.com/PaddlePaddle/Fleet>`__.

Preparation
~~~~~~~~~~~

-  [x] Install Paddle Fluid. If not already installed, please refer to
   `Beginner’s
   Guide <https://www.paddlepaddle.org.cn/documentation/docs/en/1.7/beginners_guide/index_en.html>`__.
-  [x] Master the most basic single node training method. Please refer
   to the single card training described in `Single-node
   training <https://www.paddlepaddle.org.cn/documentation/docs/en/1.5/user_guides/howto/training/single_node_en.html>`__.

Click-through rate prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we will use a simple example, click-through rate prediction task,
to illustrate how to configure Fleet API for distributed training, and
gives an example by using a single node environment to simulate the
distributed environment. The source code of the example comes from `CTR
with
Fleet <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/ctr>`__.

In order to facilitate learning, the example given here is a mixed code
of single node and multi node. You can start single node or multi node
tasks through different startup commands. For the part of obtaining data
and the logic of data preprocessing, please refer to the source code and
description of `CTR with
Fleet <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/ctr>`__.

.. code:: python

    from __future__ import print_function
    from args import parse_args
    import os
    import paddle.fluid as fluid
    import sys
    from network_conf import ctr_dnn_model_dataset
    import paddle.fluid.incubate.fleet.base.role_maker as role_maker

    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
    from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

    dense_feature_dim = 13
    sparse_feature_dim = 10000001
    batch_size = 100
    thread_num = 10
    embedding_size = 10
    args = parse_args()

    def main_function(is_local):
      # common code for local training and distributed training
      dense_input = fluid.layers.data(
        name="dense_input", shape=[dense_feature_dim], dtype='float32')

      sparse_input_ids = [
            fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1,
                              dtype="int64") for i in range(1, 27)]

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var([dense_input] + sparse_input_ids + [label])
        pipe_command = "python criteo_reader.py %d" % sparse_feature_dim
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(batch_size)
        dataset.set_thread(thread_num)

        whole_filelist = ["raw_data/part-%d" % x 
                           for x in range(len(os.listdir("raw_data")))]

        dataset.set_filelist(whole_filelist)
        loss, auc_var, batch_auc_var = ctr_dnn_model_dataset(
            dense_input, sparse_input_ids, label, embedding_size,
            sparse_feature_dim)

        exe = fluid.Executor(fluid.CPUPlace())
        def train_loop(epoch=20):
            for i in range(epoch):
                exe.train_from_dataset(program=fluid.default_main_program(),
                                       dataset=dataset,
                                       fetch_list=[auc_var],
                                       fetch_info=["auc"],
                                       debug=False)
        # local training
        def local_train():
            optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
            optimizer.minimize(loss)
            exe.run(fluid.default_startup_program())
            train_loop()

      # distributed training
        def dist_train():
            role = role_maker.PaddleCloudRoleMaker()
            fleet.init(role)
            strategy = DistributeTranspilerConfig()
            strategy.sync_mode = False
            optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
            optimizer.minimize(loss)

            if fleet.is_server():
                fleet.init_server()
                fleet.run_server()
            elif fleet.is_worker():
                fleet.init_worker()
                exe.run(fluid.default_startup_program())
                train_loop()
        if is_local:
            local_train()
        else:
            dist_train()

    if __name__ == '__main__':
        main_function(args.is_local)

-  Note: The IO method used in this example is dataset, please refer to
   `Dataset
   API <https://www.paddlepaddle.org.cn/documentation/docs/en/1.7/api/dataset.html>`__
   for specific documents and usage. For the ``train_from_dataset``
   interface, please refer to `Executor
   API <https://www.paddlepaddle.org.cn/documentation/docs/en/1.7/api/executor.html>`__.
   ``from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet``
   in this example means to introduce parameter server architecture for
   distributed training, which you can refer to `Fleet
   API <https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/user_guides/howto/training/fleet_api_howto_cn.html>`__
   for getting more about the options and examples of Fleet API.

Start command of single node training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    python train.py --is_local 1

Start command of single machine simulation distributed training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use launch\_ps, a built-in launcher of paddle, which users can
specify the number of workers and servers to start the parameter server
tasks.

.. code:: bash

    python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 train.py

The task running log can be viewed in the logs directory of the working
directory. When you can use a single machine to simulate distributed
training, you can perform true multi node distributed training. We
recommend that users refer directly to
`百度云运行分布式任务的示例 <https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/training/deploy_ctr_on_baidu_cloud_cn.html>`__.
