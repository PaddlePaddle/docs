.. _cn_api_fluid_dygraph_ParallelEnv:

ParallelEnv
-------------------------------

.. py:class:: paddle.distributed.ParallelEnv()

.. note::
    不推荐使用这个API，如果需要获取rank和world_size，建议使用 ``paddle.distributed.get_rank()`` 和  ``paddle.distributed.get_world_size()`` 。

这个类用于获取动态图模型并行执行所需的环境变量值。

动态图并行模式现在需要使用 ``paddle.distributed.launch`` 模块或者 ``paddle.distributed.spawn`` 方法启动。

代码示例
:::::::::

    .. code-block:: python

        import paddle
        import paddle.distributed as dist

        def train():
            # 1. initialize parallel environment
            dist.init_parallel_env()

            # 2. get current ParallelEnv
            parallel_env = dist.ParallelEnv()
            print("rank: ", parallel_env.rank)
            print("world_size: ", parallel_env.world_size)

            # print result in process 1:
            # rank: 1
            # world_size: 2
            # print result in process 2:
            # rank: 2
            # world_size: 2

        if __name__ == '__main__':
            # 1. start by ``paddle.distributed.spawn`` (default)
            dist.spawn(train, nprocs=2)
            # 2. start by ``paddle.distributed.launch``
            # train()

属性
::::::::::::

.. py:attribute:: rank

当前训练进程的编号。

此属性的值等于环境变量 `PADDLE_TRAINER_ID` 的值。默认值是0。

代码示例
:::::::::

    .. code-block:: python

        # execute this command in terminal: export PADDLE_TRAINER_ID=0
        import paddle.distributed as dist
        
        env = dist.ParallelEnv()
        print("The rank is %d" % env.rank)
        # The rank is 0


.. py:attribute:: world_size

参与训练进程的数量，一般也是训练所使用GPU卡的数量。

此属性的值等于环境变量 `PADDLE_TRAINERS_NUM` 的值。默认值为1。

代码示例
:::::::::

    .. code-block:: python

        # execute this command in terminal: export PADDLE_TRAINERS_NUM=4
        import paddle.distributed as dist
        
        env = dist.ParallelEnv()
        print("The world_size is %d" % env.world_size)
        # The world_size is 4


.. py:attribute:: device_id

当前用于并行训练的GPU的编号。

此属性的值等于环境变量 `FLAGS_selected_gpus` 的值。默认值是0。

代码示例
:::::::::

    .. code-block:: python

        # execute this command in terminal: export FLAGS_selected_gpus=1
        import paddle.distributed as dist
        
        env = dist.ParallelEnv()
        print("The device id are %d" % env.device_id)
        # The device id are 1


.. py:attribute:: current_endpoint

当前训练进程的终端节点IP与相应端口，形式为（机器节点IP:端口号）。例如：127.0.0.1:6170。

此属性的值等于环境变量 `PADDLE_CURRENT_ENDPOINT` 的值。默认值为空字符串""。

代码示例
:::::::::

    .. code-block:: python
            
        # execute this command in terminal: export PADDLE_CURRENT_ENDPOINT=127.0.0.1:6170
        import paddle.distributed as dist
        
        env = dist.ParallelEnv()
        print("The current endpoint are %s" % env.current_endpoint)
        # The current endpoint are 127.0.0.1:6170


.. py:attribute:: trainer_endpoints

当前任务所有参与训练进程的终端节点IP与相应端口，用于在NCCL2初始化的时候建立通信，广播NCCL ID。

此属性的值等于环境变量 `PADDLE_TRAINER_ENDPOINTS` 的值。默认值为空字符串""。

代码示例
:::::::::

    .. code-block:: python

        # execute this command in terminal: export PADDLE_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171
        import paddle.distributed as dist
        
        env = dist.ParallelEnv()
        print("The trainer endpoints are %s" % env.trainer_endpoints)
        # The trainer endpoints are ['127.0.0.1:6170', '127.0.0.1:6171']
