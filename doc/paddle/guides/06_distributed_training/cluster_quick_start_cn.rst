..  _cluster_quick_start:

分布式训练快速开始
==================

使用Fleet API进行分布式训练
---------------------------

从Paddle Fluid `Release 1.5.1 <https://github.com/PaddlePaddle/Paddle/releases/tag/v1.5.1>`_ 开始，官方推荐使用Fleet API进行分布式训练。


准备条件
^^^^^^^^


* 
  [x] 成功安装Paddle Fluid，如果尚未安装，请参考 `快速开始 <https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html>`_

* 
  [x] 学会最基本的单机训练方法，请参考 `单机训练 <https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/coding_practice/single_node.html>`_ 中描述的单机训练，进行学习

点击率预估任务
^^^^^^^^^^^^^^

本文使用一个简单的示例，点击率预估任务，来说明如何使用Fleet API进行分布式训练的配置方法，并利用单机环境模拟分布式环境给出运行示例。

为了方便学习，这里给出的示例是单机与多机混合的代码，用户可以通过不同的启动命令进行单机或多机任务的启动。

.. code-block:: python

    from __future__ import print_function
    from args import parse_args
    import os
    import sys

    import paddle
    import paddle.distributed.fleet.base.role_maker as role_maker
    import paddle.distributed.fleet as fleet

    from network_conf import ctr_dnn_model_dataset

    dense_feature_dim = 13
    sparse_feature_dim = 10000001

    batch_size = 100
    thread_num = 10
    embedding_size = 10

    args = parse_args()

    def main_function(is_local):

        # common code for local training and distributed training
        dense_input = paddle.static.data(
          name="dense_input", shape=[dense_feature_dim], dtype='float32')
        sparse_input_ids = [
              paddle.static.data(name="C" + str(i), shape=[1], lod_level=1,
                                dtype="int64") for i in range(1, 27)]

        label = paddle.static.data(name="label", shape=[1], dtype="int64")

        dataset = paddle.distributed.QueueDataset()
        dataset.init(
              batch_size=batch_size,
              thread_num=thread_num,
              input_type=0,
              pipe_command=python criteo_reader.py %d" % sparse_feature_dim,
              use_var=[dense_input] + sparse_input_ids + [label])

        whole_filelist = ["raw_data/part-%d" % x 
                           for x in range(len(os.listdir("raw_data")))]
        dataset.set_filelist(whole_filelist)

        loss, auc_var, batch_auc_var = ctr_dnn_model_dataset(
            dense_input, sparse_input_ids, label, embedding_size,
            sparse_feature_dim)

        exe = paddle.static.Executor(paddle.CPUPlace())

        def train_loop(epoch=20):
            for i in range(epoch):
                exe.train_from_dataset(program=paddle.static.default_main_program(),
                                       dataset=dataset,
                                       fetch_list=[auc_var],
                                       fetch_info=["auc"],
                                       debug=False)

        # local training
        def local_train():
            optimizer = paddle.optimizer.SGD(learning_rate=1e-4)
            optimizer.minimize(loss)
            exe.run(paddle.static.default_startup_program())
            train_loop()

      # distributed training
        def dist_train():
            role = role_maker.PaddleCloudRoleMaker()
            fleet.init(role)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.a_sync = True
            optimizer = paddle.optimizer.SGD(learning_rate=1e-4)
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
            optimizer.minimize(loss)

            if fleet.is_server():
                fleet.init_server()
                fleet.run_server()

            elif fleet.is_worker():
                fleet.init_worker()
                exe.run(paddle.static.default_startup_program())
                train_loop()

        if is_local:
            local_train()
        else:
            dist_train()

    if __name__ == '__main__':
        main_function(args.is_local)


* 说明：示例中使用的IO方法是dataset，想了解具体的文档和用法请参考 `Dataset API <https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/paddle/distributed/QueueDataset_cn.html>`_ 。

单机训练启动命令
~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train.py --is_local 1

单机模拟分布式训练的启动命令
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在单机模拟多机训练的启动命令，这里我们用到了paddle内置的一个启动器launch_ps，用户可以指定worker和server的数量进行参数服务器任务的启动

.. code-block:: bash

   python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 train.py

任务运行的日志在工作目录的logs目录下可以查看，当您能够使用单机模拟分布式训练。


