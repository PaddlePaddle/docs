使用FleetAPI进行分布式训练
====================

FleetAPI 设计说明
---------------

Fleet是PaddlePaddle分布式训练的高级API。Fleet的命名出自于PaddlePaddle，象征一个舰队中的多只双桨船协同工作。Fleet的设计在易用性和算法可扩展性方面做出了权衡。用户可以很容易从单机版的训练程序，通过添加几行代码切换到分布式训练程序。此外，分布式训练的算法也可以通过Fleet API接口灵活定义。具体的设计原理可以参考https://github.com/PaddlePaddle/Fleet/blob/develop/README.md
当前FleetAPI还处于paddle.fluid.incubate目录下，未来功能完备后会放到paddle.fluid目录中，欢迎持续关注。

快速上手示例
------------------------------
用户可以使用Fleet API轻易实现GPU多卡训练（单机多卡/多机多卡）。多卡训练在现代AI模型中非常常见，例如[Resnet50][Bert]等都是非常常见的需要多机多卡训练的模型。下面的代码示例，以一个简单的例子入手展示如何使用Fleet API进行单机多卡训练。代码示例可以参考：[quick-start]

神经网络模型的定义如下：

.. code-block:: python

   def mlp(input_x, input_y, hid_dim=128, label_dim=2):
      fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
      fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
      prediction = fluid.layers.fc(input=[fc_2], size=label_dim, act='softmax')
      cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
      avg_cost = fluid.layers.mean(x=cost)
      return avg_cost


一个简单的训练程序如下：

.. code-block:: python

   import paddle.fluid as fluid
   from nets import mlp
   from utils import gen_data

   input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
   input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

   cost = mlp(input_x, input_y)
   optimizer = fluid.optimizer.SGD(learning_rate=0.01)
   optimizer.minimize(cost)
   place = fluid.CUDAPlace(0)

   exe = fluid.Executor(place)
   exe.run(fluid.default_startup_program())
   step = 1001
   for i in range(step):
       exe.run(feed=gen_data())


If you want to use high performance chip to do distributed training, such as distributed GPU training, **Fleet API** will help you by adding less than 10 lines of code, source code of this example is in examples/quick-start/collective_trainer.py
如果用户想使用高性能芯片，例如GPU多卡进行训练，使用**Fleet API**可以在增加少量代码的情况下实现。

.. code-block:: python

   import paddle.fluid as fluid
   from utils import gen_data
   from nets import mlp
   from paddle.fluid.incubate.fleet.collective import fleet
   from paddle.fluid.incubate.fleet.base import role_maker

   input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
   input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

   cost = mlp(input_x, input_y)
   optimizer = fluid.optimizer.SGD(learning_rate=0.01)

   role = role_maker.PaddleCloudRoleMaker()
   fleet.init(role)
   optimizer = fleet.distributed_optimizer(optimizer)
   optimizer.minimize(cost)

   place = fluid.CUDAPlace(0)

   exe = fluid.Executor(place)
   exe.run(fluid.default_startup_program())
   step = 1001
   for i in range(step):
       exe.run(feed=gen_data())


在单机运行多卡程序的执行命令如下：

.. code-block:: python

   python -m paddle.distributed.launch collective_trainer.py



FleetAPI 接口说明
------------------------------
.. csv-table::
   :header: "接口", "说明"

   "init", "fleet初始化，需要在使用fleet其他接口前先调用，用于定义多机的环境配置"
   "distributed_optimizer", "fleet多机训练策略优化，接收一个标准Optimizer及相应的多机运行策略，fleet会根据优化策略进行优化"
   "init_server", "fleet加载model_dir中保存的模型相关参数进行parameter server的初始化"
   "run_server", "fleet启动parameter server服务"
   "init_worker", "fleet初始化当前worker运行环境"
   "is_worker", "判断当前节点是否是Worker节点，是则返回True，否则返回False"
   "is_server", "判断当前节点是否是Server节点，是则返回True，否则返回False"
   "save_inference_model", "fleet保存预测相关的模型及参数，参数及用法参考 code:`fluid.io.save_inference_model`"
   "save_persistables", "fleet保存多机模型参数，参数及用法参考 code:`fluid.io.save_persistables`"


FleetAPI 一般训练步骤
------------------------------

通过import引入需要使用的模式
++++++++++++++++++

使用parameter server方式的训练：

.. code-block:: python

    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet


初始化
++++++++++++++++++
Fleet使用 code:`fleet.init(role_maker=None)` 进行初始化

当用户不指定role_maker, 则Fleet默认用户使用MPI环境，会采用MPISymetricRoleMaker.

如果用户使用非MPI环境，则需要通过UserDefinedRoleMaker自行定义执行环境。
例如：

.. code-block:: python

    role = UserDefinedRoleMaker(current_id=0,
                     role=Role.WORKER,
                     worker_num=3,
                     server_endpoints=["127.0.0.1:6001","127.0.0.1:6002"])
    fleet.init(role_maker=role)


分布式策略及多机配置
++++++++++++++++

对于Transpiler模式，需要使用 DistributeTranspilerConfig 指定多机配置。
Fleet需要在用户定义的optimizer之上装饰 code:`fleet.distributed_optimizer` 来完成多机分布式策略的配置。

.. csv-table::
   :header: "接口", "说明"

   "sync_mode", "Fleet可以支持同步训练或异步训练， 默认会生成同步训练的分布式程序，通过指定 :code:`sync_mode=False` 参数即可生成异步训练的程序"
   "split_method", "指定参数在parameter server上的分布方式, 默认使用 `RoundRobin`, 也可选`HashName`"
   "slice_var_up", "指定是否将较大（大于8192个元素）的参数切分到多个parameter server以均衡计算负载，默认为开启"


例如：

.. code-block:: python

    config = DistributeTranspilerConfig()
    config.sync_mode = True
   
    # build network
    # ...
    avg_cost = model()
    
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    # 加入 fleet distributed_optimizer 加入分布式策略配置及多机优化
    optimizer = fleet.distributed_optimizer(optimizer, config)
    optimizer.minimize(avg_cost)


具体训练流程
++++++++++++++++

.. code-block:: python

    # 启动server
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
 
    # 启动worker
    if fleet.is_worker():
        # 初始化worker配置
        fleet.init_worker()
    
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        train_reader = paddle.batch(fake_reader(), batch_size=24)
    
        exe.run(fleet.startup_program)
    
        PASS_NUM = 10
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                avg_loss_value, auc_value, auc_batch_value = \ 
                    exe.run(fleet.main_program, feed=feeder.feed(data), fetch_list=[avg_cost, auc, auc_batch])
                print("Pass %d, cost = %f, auc = %f, batch_auc = %f" % (pass_id, avg_loss_value, auc_value, auc_batch_value))
        # 通知server，当前节点训练结束
        fleet.stop_worker()


