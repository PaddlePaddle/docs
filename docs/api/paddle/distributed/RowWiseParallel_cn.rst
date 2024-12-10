.. _cn_api_paddle_distributed_RowWiseParallel:

RowWiseParallel
-------------------------------

.. py:class:: paddle.distributed.RowWiseParallel(is_input_parallel=True)

按行切分策略是模型并行中的一个并行策略。本策略会尝试对标识 Layer 的 ``weight`` （如有）在第一维切分，
本策略是针对 ``paddle.nn.Linear`` 与 ``paddle.nn.Embedding`` 设计，
如果标识的 Layer 不属于这两个类，本策略会尝试切分 Layer 中的 ``weight`` （如有）。


.. note::
    ``weight`` 需要有两个纬度。


参数
:::::::::
    - **is_input_parallel** (bool，可选) - 标识 Layer 的输入是否为局部张量。如果 Layer 的输入为全局张量，额外的计算会被引入。默认为 True，即标识 Layer 的输入为局部变量。


**代码示例**

COPY-FROM: paddle.distributed.RowWiseParallel
