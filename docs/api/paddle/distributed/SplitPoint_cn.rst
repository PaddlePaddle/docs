.. _cn_api_paddle_distributed_SplitPoint:

SplitPoint
-------------------------------

.. py:class:: paddle.distributed.SplitPoint

用于流水线并行下切分位置的确认。目前支持 ``BEGINNING`` 和 ``END`` 。

``BEGINNING`` 表明在标识的 Layer 之前进行切分。

``END`` 表明在标识的 Layer 之后进行切分。


**代码示例**

COPY-FROM: paddle.distributed.SplitPoint
