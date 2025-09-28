.. _cn_api_paddle_distributed_ColWiseParallel:

ColWiseParallel
-------------------------------

.. py:class:: paddle.distributed.ColWiseParallel(gather_output=False)

按列切分策略是模型并行中的一个并行策略。本策略会尝试对标识 Layer 的 ``weight`` （如有）在第二维切分，
对标识 Layer 的 ``bias`` （如有）在第一维度切分。本策略是针对 ``paddle.nn.Linear`` 与 ``paddle.nn.Embedding`` 设计，
如果标识的 Layer 不属于这两个类，本策略会尝试切分 Layer 中的 ``weight`` 与 ``bias`` （如有）。


.. note::
    ``weight`` 需要有两个纬度。

    ``bias`` 需要有一个纬度。


参数
:::::::::
    - **gather_output** (bool，可选) - 是否将标识 Layer 的输出从局部张量变为全局张量。如果选择将局部张量变为全局张量，额外的通讯会被引入。默认为 False，即保持标识 Layer 的输出为局部变量。


**代码示例**

COPY-FROM: paddle.distributed.ColWiseParallel
