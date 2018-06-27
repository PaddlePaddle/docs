.. _cluster_howto

Fluid分布式训练使用手册
====================

分布式训练基本思想
---------------

分布式深度学习训练通常分为两种并行化方法：数据并行，模型并行，参考下图：

.. image:: src/parallelism.png

在模型并行方式下，模型的层和参数将被分布在多个节点上，模型在一个mini-batch的前向和反向训练中，将经过多次跨
节点之间的通信。每个节点只保存整个模型的一部分；在数据并行方式下，每个节点保存有完整的模型的层和参数，每个节点
独自完成前向和反向计算，然后完成梯度的聚合并同步的更新所有节点上的参数。Fluid目前版本仅提供数据并行方式，另外
诸如模型并行的特例实现（超大稀疏模型训练）功能将在后续的文档中予以说明。

在数据并行模式的训练中，Fluid使用了两种通信模式，用于应对不同训练任务对分布式训练的要求，分别为RPC通信和Collective
通信。其中RPC通信方式使用 `gRPC<https://github.com/grpc/grpc/>`_ ，Collective通信方式使用
 `NCCL2<https://developer.nvidia.com/nccl)`_ 。下面是一个RPC通信和Collective通信的横向对比：

.. csv-table:: 通信对比
   :header: "Feature", "Coolective", "RPC"

   "Ring-Based通信", "Yes", "No"
   "异步训练", "Yes", "Yes"
   "分布式模型", "No", "Yes"
   "容错训练", "No", "Yes"
   "性能", "Faster", "Fast"

- RPC通信方式的结构：

  .. image:: src/dist_train_pserver.png
     :width: 500px

- NCCL2通信方式的结构：

  .. image:: src/dist_train_nccl2.png
     :width: 500px


使用parameter server方式的训练
---------------------------

使用"trainer" API，程序可以自动的通过识别环境变量决定是否已分布式方式执行，需要在您的分布式环境中配置的环境变量包括：

.. csv-table:: pserver模式环境变量
   :header: "环境变量", "说明"

   "PADDLE_TRAINING_ROLE", "当前进程的角色，可以是PSERVER或TRAINER"
   "PADDLE_PSERVER_PORT", "parameter使用的端口"
   "PADDLE_PSERVER_IPS", "parameter server的IP地址列表，用逗号分开"
   "PADDLE_TRAINERS", "分布式任务中trainer节点的个数"
   "PADDLE_CURRENT_IP", "当前节点的IP"
   "PADDLE_TRAINER_ID", "trainer节点的id，从0~n-1，不能有重复"

使用更加底层的"transpiler" API可以提供自定义的分布式训练的方法，比如可以在同一台机器上，启动多个pserver和trainer
进行训练，使用底层API的方法可以参考下面的样例代码：

.. code:: python

   role = "PSERVER"
   trainer_id = 0
   pserver_endpoints = "127.0.0.1:6170,127.0.0.1:6171"
   current_endpoint = "127.0.0.1:6170"
   trainers = 4
   t = fluid.DistributeTranspiler()
   t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
   if role == "PSERVER":
       pserver_prog = t.get_pserver_program(current_endpoint)
       pserver_startup = t.get_startup_program(current_endpoint,
                                               pserver_prog)
       exe.run(pserver_startup)
       exe.run(pserver_prog)
   elif role == "TRAINER":
       train_loop(t.get_trainer_program())


使用NCCL2通信方式的训练
--------------------

注NCCL2模式目前仅支持trainer API，NCCL2方式并没有很多可选项，也没有"transpiler"，所以并没有底层API。
使用NCCL2方式同样需要配置每个节点的环境变量，此处与parameter server模式有所不同，并不需要启动独立的
parameter server的进程，只需要启动多个trainer进程即可：


.. csv-table:: pserver模式环境变量
   :header: "环境变量", "说明"

   "PADDLE_TRAINER_IPS", "所有Trainer节点的IP列表，用逗号分隔"
   "PADDLE_TRAINER_ID", "trainer节点的id，从0~n-1，不能有重复"
   "PADDLE_PSERVER_PORT", "一个端口，用于在NCCL2初始化时，广播NCCL ID"
   "PADDLE_CURRENT_IP", "当前节点的IP"

