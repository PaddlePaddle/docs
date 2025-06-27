.. _cn_api_paddle_distributed_SequenceParallelDisable:

SequenceParallelDisable
-------------------------------

.. py:class:: paddle.distributed.SequenceParallelDisable(need_transpose=True)

在序列并行的区间内，不对标识的 Layer 进行序列并行。


参数
:::::::::
    - **need_transpose** (bool，可选) - 是否在策略中进行 ``transpose`` 操作。
      如果为 True，本策略会将形状为 ``[s/mp, b, h]`` 的输入张量变为形状为 ``[b, s, h]`` 的张量来传递给标识 Layer 进行运算，
      并将标识 Layer 形状为 ``[b, s, h]`` 的输出张量变为形状为 ``[s/mp, b, h]`` 的张量来输出。
      如果为 False，本策略会将形状为 ``[s/mp, b, h]`` 的输入变为形状为 ``[s, b, h]`` 的张量来传递给标识 Layer 进行运算，
      并将标识 Layer 形状为 ``[s, b, h]`` 的输出张量变为形状为 ``[s/mp, b, h]`` 的张量来输出。
      默认为 True。


**代码示例**

COPY-FROM: paddle.distributed.SequenceParallelDisable
