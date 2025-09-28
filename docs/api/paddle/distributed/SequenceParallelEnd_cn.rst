.. _cn_api_paddle_distributed_SequenceParallelEnd:

SequenceParallelEnd
-------------------------------

.. py:class:: paddle.distributed.SequenceParallelEnd(need_transpose=True)

标识序列并行结束的策略，该策略应该标识在序列并行结束之后的最后一个 Layer 上。


.. note::
    不要将应该进行序列并行的 Layer 用此策略标识。


参数
:::::::::
    - **need_transpose** (bool，可选) - 是否在策略中进行 ``transpose`` 操作。
      如果为 True，本策略会讲形状为 ``[s/mp, b, h]`` 的输入张量变为形状为 ``[b, s, h]`` 的输出张量。
      反之，本策略会将形状为 ``[s/mp, b, h]`` 的输入张量变为形状为 ``[s, b, h]`` 的输出张量。默认为 True。


**代码示例**

COPY-FROM: paddle.distributed.SequenceParallelEnd
