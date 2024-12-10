.. _cn_api_paddle_distributed_SequenceParallelEnable:

SequenceParallelEnable
-------------------------------

.. py:class:: paddle.distributed.SequenceParallelEnable()

对标识 Layer 进行序列并行策略。

.. note::
    被标识的 Layer 的输入张量的形状应该为 ``[b, s, h]``。

**代码示例**

COPY-FROM: paddle.distributed.SequenceParallelEnable
