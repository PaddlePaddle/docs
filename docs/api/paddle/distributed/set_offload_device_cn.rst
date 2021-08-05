.. _cn_api_distributed_set_offload_device:

shard_op
-------------------------------


.. py:function:: paddle.distributed.set_offload_device(x, device)

设置输入Tensor `x` 被放置的设备。

参数
:::::::::
    - x (Tensor) - 待操作的输入Tensor。
    - device (str) - 放置输入Tensor的目标设备，如'cpu', 'gpu:0'等。

返回
:::::::::
Tensor: 输入Tensor `x` 自身。

代码示例
:::::::::
.. code-block:: python

    import numpy as np
    import paddle
    import paddle.distributed as dist

    paddle.enable_static()

    x = paddle.ones([4, 6])
    dist.set_offload_device(x, 'cpu')
