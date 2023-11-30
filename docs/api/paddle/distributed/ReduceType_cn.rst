.. _cn_api_paddle_distributed_ReduceType:

ReduceType
-------------------------------

.. py:class:: paddle.distributed.ReduceType

指定分布式 Tensor 在 Partial 状态下规约操作的类型，必须是下述值之一：

    ReduceType.kRedSum

    ReduceType.kRedMax

    ReduceType.kRedMin

    ReduceType.kRedProd

    ReduceType.kRedAvg

    ReduceType.kRedAny

    ReduceType.kRedAll

代码示例
:::::::::

COPY-FROM: paddle.distributed.ReduceType
