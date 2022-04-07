.. _cn_api_distributed_ParallelMode:

ParallelMode

-------------------------------

.. py:class:: paddle.distributed.ParallelMode

以下是目前支持的并行模式：
    - 数据并行：将输入数据分发在不同的设备。
    - 张量并行：将网络中的张量切分到不同的设备。
    - 流水线并行：将网络的不同层切分到不同的设备。
    - 切片并行：将模型参数、参数梯度、参数对应的优化器状态切分到每一个设备。

代码示例
::::::::::::

.. code-block:: python

    import paddle
    parallel_mode = paddle.distributed.ParallelMode
    print(parallel_mode.DATA_PARALLEL)  # 0
