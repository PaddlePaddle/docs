.. _cn_api_fluid_dygraph_ParallelEnv:

ParallelEnv
-------------------------------

.. py:class:: paddle.distributed.ParallelEnv()

.. note::
    不推荐使用这个 API，如果需要获取 rank 和 world_size，建议使用 ``paddle.distributed.get_rank()`` 和  ``paddle.distributed.get_world_size()`` 。

这个类用于获取动态图模型并行执行所需的环境变量值。

动态图并行模式现在需要使用 ``paddle.distributed.launch`` 模块或者 ``paddle.distributed.spawn`` 方法启动。

代码示例
:::::::::

COPY-FROM: paddle.distributed.ParallelEnv

属性
::::::::::::
rank
'''''''''

当前训练进程的编号。

此属性的值等于环境变量 `PADDLE_TRAINER_ID` 的值。默认值是 0。

**代码示例**

COPY-FROM: paddle.distributed.ParallelEnv.rank

world_size
'''''''''

参与训练进程的数量，一般也是训练所使用 GPU 卡的数量。

此属性的值等于环境变量 `PADDLE_TRAINERS_NUM` 的值。默认值为 1。

**代码示例**

COPY-FROM: paddle.distributed.ParallelEnv.world_size

device_id
'''''''''

当前用于并行训练的 GPU 的编号。

此属性的值等于环境变量 `FLAGS_selected_gpus` 的值。默认值是 0。

**代码示例**

COPY-FROM: paddle.distributed.ParallelEnv.device_id

current_endpoint
'''''''''

当前训练进程的终端节点 IP 与相应端口，形式为（机器节点 IP:端口号）。例如：127.0.0.1:6170。

此属性的值等于环境变量 `PADDLE_CURRENT_ENDPOINT` 的值。默认值为空字符串""。

**代码示例**

COPY-FROM: paddle.distributed.ParallelEnv.current_endpoint

trainer_endpoints
'''''''''

当前任务所有参与训练进程的终端节点 IP 与相应端口，用于在 NCCL2 初始化的时候建立通信，广播 NCCL ID。

此属性的值等于环境变量 `PADDLE_TRAINER_ENDPOINTS` 的值。默认值为空字符串""。

**代码示例**

COPY-FROM: paddle.distributed.ParallelEnv.trainer_endpoints
