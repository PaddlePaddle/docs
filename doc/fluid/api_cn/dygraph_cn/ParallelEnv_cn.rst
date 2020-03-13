.. _cn_api_fluid_dygraph_ParallelEnv:

ParallelEnv
-------------------------------

.. py:class:: paddle.fluid.dygraph.ParallelEnv()

**注意：**
  **这个类的曾用名为 Env， 这个旧的名字会被废弃，请使用新的类名 ParallelEnv。**

这个类用于获取动态图模型并行执行所需的环境变量值。

动态图并行模式现在需要使用 `paddle.distributed.launch` 模块启动，所需的环境变量默认由 `paddle.distributed.launch` 模块自动配置。

ParallelEnv通常需要和 `fluid.dygraph.DataParallel` 一起使用，用于配置动态图并行执行。

**示例代码：**
    .. code-block:: python

        # 这个示例需要由paddle.distributed.launch启动, 用法为:
        #   python -m paddle.distributed.launch --selected_gpus=0,1 example.py
        # 脚本example.py中的代码是下面这个示例.

        import numpy as np
        import paddle.fluid as fluid
        import paddle.fluid.dygraph as dygraph
        from paddle.fluid.optimizer import AdamOptimizer
        from paddle.fluid.dygraph.nn import Linear
        from paddle.fluid.dygraph.base import to_variable

        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):

            # 准备数据并行的环境
            strategy=dygraph.prepare_context()

            linear = Linear(1, 10, act="softmax")
            adam = fluid.optimizer.AdamOptimizer()

            # 配置模型为并行模型
            linear = dygraph.DataParallel(linear, strategy)

            x_data = np.random.random(size=[10, 1]).astype(np.float32)
            data = to_variable(x_data)

            hidden = linear(data)
            avg_loss = fluid.layers.mean(hidden)

            # 根据参与训练GPU卡的数量对loss值进行缩放
            avg_loss = linear.scale_loss(avg_loss)

            avg_loss.backward()

            # 收集各个GPU卡上的梯度值
            linear.apply_collective_grads()

            adam.minimize(avg_loss)
            linear.clear_gradients()

属性
::::::::::::

.. py:attribute:: nranks

参与训练进程的数量，一般也是训练所使用GPU卡的数量。

此属性的值等于环境变量 `PADDLE_TRAINERS_NUM` 的值。默认值为1。

**示例代码**
    .. code-block:: python

        # 在Linux环境，提前执行此命令: export PADDLE_TRAINERS_NUM=4
        import paddle.fluid as fluid
        
        env = fluid.dygraph.ParallelEnv()
        print("The nranks is %d" % env.nranks)
        # The nranks is 4


.. py:attribute:: local_rank

当前训练进程的编号。

此属性的值等于环境变量 `PADDLE_TRAINER_ID` 的值。默认值是0。

**示例代码**
    .. code-block:: python

        # 在Linux环境，提前执行此命令: export PADDLE_TRAINER_ID=0
        import paddle.fluid as fluid
        
        env = fluid.dygraph.ParallelEnv()
        print("The local rank is %d" % env.local_rank)
        # The local rank is 0


.. py:attribute:: dev_id

当前用于并行训练的GPU的编号。

此属性的值等于环境变量 `FLAGS_selected_gpus` 的值。默认值是0。

**示例代码**
    .. code-block:: python

        # 在Linux环境，提前执行此命令: export FLAGS_selected_gpus=1
        import paddle.fluid as fluid
        
        env = fluid.dygraph.ParallelEnv()
        print("The device id are %d" % env.dev_id)
        # The device id are 1


.. py:attribute:: current_endpoint

当前训练进程的终端节点IP与相应端口，形式为（机器节点IP:端口号）。例如：127.0.0.1:6170。

此属性的值等于环境变量 `PADDLE_CURRENT_ENDPOINT` 的值。默认值为空字符串""。

**示例代码**
    .. code-block:: python
            
        # 在Linux环境，提前执行此命令: export PADDLE_CURRENT_ENDPOINT=127.0.0.1:6170
        import paddle.fluid as fluid
        
        env = fluid.dygraph.ParallelEnv()
        print("The current endpoint are %s" % env.current_endpoint)
        # The current endpoint are 127.0.0.1:6170


.. py:attribute:: trainer_endpoints

当前任务所有参与训练进程的终端节点IP与相应端口，用于在NCCL2初始化的时候建立通信，广播NCCL ID。

此属性的值等于环境变量 `PADDLE_TRAINER_ENDPOINTS` 的值。默认值为空字符串""。

**示例代码**
    .. code-block:: python

        # 在Linux环境，提前执行此命令: export PADDLE_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171
        import paddle.fluid as fluid
        
        env = fluid.dygraph.ParallelEnv()
        print("The trainer endpoints are %s" % env.trainer_endpoints)
        # The trainer endpoints are ['127.0.0.1:6170', '127.0.0.1:6171']