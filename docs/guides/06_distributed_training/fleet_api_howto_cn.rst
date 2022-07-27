
使用 FleetAPI 进行分布式训练
==========================

FleetAPI 设计说明
-----------------

Fleet 是 PaddlePaddle 分布式训练的高级 API。Fleet 的命名出自于 PaddlePaddle，象征一个舰队中的多只双桨船协同工作。Fleet 的设计在易用性和算法可扩展性方面做出了权衡。用户可以很容易从单机版的训练程序，通过添加几行代码切换到分布式训练程序。此外，分布式训练的算法也可以通过 Fleet
API 接口灵活定义。

Fleet API 快速上手示例
---------------------

下面会针对 Fleet
API 最常见的两种使用场景，用一个模型做示例，目的是让用户有快速上手体验的模板。


*
  假设我们定义 MLP 网络如下：

  .. code-block:: python

     import paddle

     def mlp(input_x, input_y, hid_dim=128, label_dim=2):
       fc_1 = paddle.static.nn.fc(input=input_x, size=hid_dim, act='tanh')
       fc_2 = paddle.static.nn.fc(input=fc_1, size=hid_dim, act='tanh')
       prediction = paddle.static.nn.fc(input=[fc_2], size=label_dim, act='softmax')
       cost = paddle.static.nn.cross_entropy(input=prediction, label=input_y)
       avg_cost = paddle.static.nn.mean(x=cost)
       return avg_cost

*
  定义一个在内存生成数据的 Reader 如下：

  .. code-block:: python

     import numpy as np

     def gen_data():
         return {"x": np.random.random(size=(128, 32)).astype('float32'),
                 "y": np.random.randint(2, size=(128, 1)).astype('int64')}

*
  单机 Trainer 定义

  .. code-block:: python

     import paddle
     from nets import mlp
     from utils import gen_data

     input_x = paddle.static.data(name="x", shape=[None, 32], dtype='float32')
     input_y = paddle.static.data(name="y", shape=[None, 1], dtype='int64')

     cost = mlp(input_x, input_y)
     optimizer = paddle.optimizer.SGD(learning_rate=0.01)
     optimizer.minimize(cost)
     place = paddle.CUDAPlace(0)

     exe = paddle.static.Executor(place)
     exe.run(paddle.static.default_startup_program())
     step = 1001
     for i in range(step):
       cost_val = exe.run(feed=gen_data(), fetch_list=[cost.name])
       print("step%d cost=%f" % (i, cost_val[0]))

*
  Parameter Server 训练方法

  参数服务器方法对于大规模数据，简单模型的并行训练非常适用，我们基于单机模型的定义给出使用 Parameter Server 进行训练的示例如下：

  .. code-block:: python

    import paddle
    paddle.enable_static()

    import paddle.distributed.fleet.base.role_maker as role_maker
    import paddle.distributed.fleet as fleet

    from nets import mlp
    from utils import gen_data

    input_x = paddle.static.data(name="x", shape=[None, 32], dtype='float32')
    input_y = paddle.static.data(name="y", shape=[None, 1], dtype='int64')

    cost = mlp(input_x, input_y)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01)

    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.a_sync = True

    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(cost)

    if fleet.is_server():
      fleet.init_server()
      fleet.run_server()

    elif fleet.is_worker():
      place = paddle.CPUPlace()
      exe = paddle.static.Executor(place)
      exe.run(paddle.static.default_startup_program())

      step = 1001
      for i in range(step):
        cost_val = exe.run(
            program=paddle.static.default_main_program(),
            feed=gen_data(),
            fetch_list=[cost.name])
        print("worker_index: %d, step%d cost = %f" %
             (fleet.worker_index(), i, cost_val[0]))

*
  Collective 训练方法

  Collective Training 通常在 GPU 多机多卡训练中使用，一般在复杂模型的训练中比较常见，我们基于上面的单机模型定义给出使用 Collective 方法进行分布式训练的示例如下：

  .. code-block:: python

     import paddle
     paddle.enable_static()

     import paddle.distributed.fleet.base.role_maker as role_maker
     import paddle.distributed.fleet as fleet

     from nets import mlp
     from utils import gen_data

     input_x = paddle.static.data(name="x", shape=[None, 32], dtype='float32')
     input_y = paddle.static.data(name="y", shape=[None, 1], dtype='int64')

     cost = mlp(input_x, input_y)
     optimizer = paddle.optimizer.SGD(learning_rate=0.01)
     role = role_maker.PaddleCloudRoleMaker(is_collective=True)
     fleet.init(role)

     optimizer = fleet.distributed_optimizer(optimizer)
     optimizer.minimize(cost)
     place = paddle.CUDAPlace(0)

     exe = paddle.static.Executor(place)
     exe.run(paddle.static.default_startup_program())

     step = 1001
     for i in range(step):
       cost_val = exe.run(
           program=paddle.static.default_main_program(),
           feed=gen_data(),
           fetch_list=[cost.name])
       print("worker_index: %d, step%d cost = %f" %
            (fleet.worker_index(), i, cost_val[0]))


Fleet API 相关的接口说明
-----------------------

Fleet API 接口
^^^^^^^^^^^^^


* init(role_maker=None)

  * fleet 初始化，需要在使用 fleet 其他接口前先调用，用于定义多机的环境配置

* is_worker()

  * Parameter Server 训练中使用，判断当前节点是否是 Worker 节点，是则返回 True，否则返回 False

* is_server(model_dir=None)

  * Parameter Server 训练中使用，判断当前节点是否是 Server 节点，是则返回 True，否则返回 False

* init_server()

  * Parameter Server 训练中，fleet 加载 model_dir 中保存的模型相关参数进行 parameter
    server 的初始化

* run_server()

  * Parameter Server 训练中使用，用来启动 server 端服务

* init_worker()

  * Parameter Server 训练中使用，用来启动 worker 端服务

* stop_worker()

  * 训练结束后，停止 worker

* distributed_optimizer(optimizer, strategy=None)

  * 分布式优化算法装饰器，用户可带入单机 optimizer，并配置分布式训练策略，返回一个分布式的 optimizer

RoleMaker
^^^^^^^^^


*
  PaddleCloudRoleMaker


  *
    描述：PaddleCloudRoleMaker 是一个高级封装，支持使用 paddle.distributed.launch 或者 paddle.distributed.launch_ps 启动脚本

  *
    Parameter Server 训练示例：

    .. code-block:: python

       import paddle
       paddle.enable_static()

       import paddle.distributed.fleet.base.role_maker as role_maker
       import paddle.distributed.fleet as fleet

       role = role_maker.PaddleCloudRoleMaker()
       fleet.init(role)

  *
    启动方法：

    .. code-block:: python

       python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 trainer.py

  *
    Collective 训练示例：

    .. code-block:: python

       import paddle
       paddle.enable_static()

       import paddle.distributed.fleet.base.role_maker as role_maker
       import paddle.distributed.fleet as fleet

       role = role_maker.PaddleCloudRoleMaker(is_collective=True)
       fleet.init(role)

  *
    启动方法：

    .. code-block:: python

        python -m paddle.distributed.launch trainer.py

*
  UserDefinedRoleMaker


  *
    描述：用户自定义节点的角色信息，IP 和端口信息

  *
    示例：

    .. code-block:: python

       import paddle
       paddle.enable_static()

       import paddle.distributed.fleet.base.role_maker as role_maker
       import paddle.distributed.fleet as fleet

       role = role_maker.UserDefinedRoleMaker(
           current_id=0,
           role=role_maker.Role.SERVER,
           worker_num=2,
           server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"])

       fleet.init(role)
