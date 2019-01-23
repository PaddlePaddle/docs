..  _cluster_quick_start_en:

Quick Start with Distributed Training
==========================

Preparation
--------------------
In this article, we'll show you how to quickly start a PaddlePaddle distributed training task in a cluster. Before you start, do some preparatory work as follows:

1. Prepare a connected training cluster. Here we use 4 training nodes with format ``*.paddlepaddle.com`` to represent the host name of the node. You can modify it according to the actual situation.

2. Make sure you have read :ref:`install_steps` before you start and can run PaddlePaddle on all nodes of the cluster.

Example code
-------------

Let's use a very simple linear regression model as an example to explain how to start a distributed training task with 2 pserver server nodes and 2 trainer nodes. You can save this code as ``dist_train.py`` .

.. code:: python


    import os
    import paddle
    import paddle.fluid as fluid

    # train reader
    BATCH_SIZE = 20
    EPOCH_NUM = 30
    BATCH_SIZE = 8

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    def train():
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)

        loss = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_loss = fluid.layers.mean(loss)
        opt = fluid.optimizer.SGD(learning_rate=0.001)
        opt.minimize(avg_loss)

        place = fluid.CPUPlace()
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)

        # fetch distributed training environment setting
        training_role = os.getenv("PADDLE_TRAINING_ROLE", None)
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port

        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id = trainer_id,
            pservers = pserver_endpoints,
            trainers = trainers)

        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
            exe.run(startup_prog)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            trainer_prog = t.get_trainer_program()
            exe.run(fluid.default_startup_program())

            for epoch in range(EPOCH_NUM):
                for batch_id, batch_data in enumerate(train_reader()):
                    avg_loss_value, = exe.run(trainer_prog,
                                          feed=feeder.feed(batch_data),
                                          fetch_list=[avg_loss])
                    if (batch_id + 1) % 10 == 0:
                        print("Epoch: {0}, Batch: {1}, loss: {2}".format(
                            epoch, batch_id, avg_loss_value[0]))
            # destory the resource of current trainer node in pserver server node
            exe.close()
        else:
            raise AssertionError("PADDLE_TRAINING_ROLE should be one of [TRAINER, PSERVER]")

    train()


Environment Variables
------------------------------------

When starting a distributed training task, different environment variables are used to represent different node roles, details as follows:

.. list-table::
  :header-rows: 1

  * - Environment Variable
    - Data Type 
    - Example 
    - Description
  * - :code:`PADDLE_TRAINING_ROLE`
    - str 
    - :code:`PSERVER,TRANERR`
    - role of current training node
  * - :code:`PADDLE_PSERVER_IPS`
    - str 
    - :code:`ps0.paddlepaddle.com, ps1.paddlepaddle.com`
    - The IP addresses or hostnames of all pserver nodes in the distributed training task, separated by ","
  * - :code:`PADDLE_PSERVER_PORT`
    - int 
    - 6174 
    - port that the pserver process listens to
  * - :code:`PADDLE_TRAINERS`
    - int
    - 2 
    - Number of trainer nodes in a distributed training task
  * - :code:`PADDLE_CURRENT_IP`
    - str 
    - :code:`ps0.paddlepaddle.com`
    - IP address or hostname of the current pserver node
  * - :code:`PADDLE_TRAINER_ID`
    - str 
    - 0 
    - ID of the current trainer node (unique), in the range of [0, PADDLE_TRAINERS)

**Note:** Environment variables are just a way to get runtime information. In practical tasks, you can use command line parameters to obtain runtime information.

API related to Distributed Training
---------------------------------

DistributeTranspiler
~~~~~~~~~~~~~~~~~~~~~~

The machines in distributed training tasks based on the pserver-trainer architecture are divided into two roles: Parameter Server (pserver) and trainer. In Fluid, users only need to configure the network configuration required for single node training. The ``DistributeTranspiler`` module automatically modifies the single-node network settings into settings on which pserver and trainer needs to run based on the role of current training node:

.. code:: python

  t = fluid.DistributeTranspiler()
  t.transpile(
    trainer_id = trainer_id,
    pservers = pserver_endpoints,
    trainers = trainers)
  if PADDLE_TRAINING_ROLE == "TRAINER":
    # fetch the pserver program and execute it
    trainer_prog = t.get_trainer_program()
    ...

  elif PADDLE_TRAINER_ROLE == "PSERVER":
    # fetch the trainer program and execute it
    pserver_prog = t.get_pserver_program(current_endpoint)
    ...


Exe.close()
~~~~~~~~~~~~~~


The status information of all trainer nodes is saved in the pserver node. When trainer finishes training, ``exe.close()`` should be called to notify all PServer nodes to release the resources of the current Trainer nodes:

.. code:: python

  exe = fluid.Executor(fluid.CPUPlace())
  # training process ...
  exe.close() # notify PServer to destory the resource


Start a Distributed Training Task
----------------------------------

.. list-table::
   :header-rows: 1


   * - Start Node 
     - Start Command 
     - Description
   * - ps0.paddlepaddle.com 
     - :code:`PADDLE_TRAINING_ROLE=PSERVER PADDLE_CURRENT_IP=ps0.paddlepaddle.com PADDLE_PSERVER_IPS=ps0.paddlepaddle.com, ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - Start pserver node
   * - ps1.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=PSERVER PADDLE_CURRENT_IP=ps1.paddlepaddle.com PADDLE_PSERVER_IPS=ps0.paddlepaddle.com, ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - Start pserver node
   * - trainer0.paddlepaddle.com       
     - :code:`PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_IPS=ps0.paddlepaddle.com, ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_TRAINER_ID=0 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - Start the number 0 Trainer Node 
   * - trainer1.paddlepaddle.com       
     - :code:`PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_IPS=ps0.paddlepaddle.com, ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_TRAINER_ID=1 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - Start the number 1 trainer node
