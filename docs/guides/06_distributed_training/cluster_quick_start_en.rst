Quick start for distributed training
====================================

Distributed training with Fleet API
-----------------------------------

Since PaddlePaddle `Release
1.5.1 <https://github.com/PaddlePaddle/Paddle/releases/tag/v1.5.1>`__,
it is officially recommended to use the Fleet API for distributed
training.

Preparation
~~~~~~~~~~~

-  [x] Install PaddlePaddle. If not already installed, please refer to
   `Beginner’s
   Guide <https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html>`__.
-  [x] Master the most basic single node training method. Please refer
   to the single card training described in `Single-node
   training <https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/coding_practice/single_node_en.html>`__.

Click-through rate prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we will use a simple example, click-through rate prediction task,
to illustrate how to configure Fleet API for distributed training, and
gives an example by using a single node environment to simulate the
distributed environment.

In order to facilitate learning, the example given here is a mixed code
of single node and multi node. You can start single node or multi node
tasks through different startup commands.

.. code-block:: python

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


-  Note: The IO method used in this example is dataset, please refer to
   `Dataset
   API <https://www.paddlepaddle.org.cn/documentation/docs/en/2.0-rc1/api/paddle/distributed/QueueDataset_en.html>`__
   for specific documents and usage.

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
training, you can perform true multi node distributed training.
