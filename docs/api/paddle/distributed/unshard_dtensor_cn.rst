.. _cn_api_paddle_distributed_unshard_dtensor:

unshard_dtensor
-------------------------------

.. py:function:: paddle.distributed.unshard_dtensor(dist_tensor)

将带有分布式信息的分布式 Tensor 转换为普通 Tensor。


参数
:::::::::

    - **dist_tensor** (paddle.Tensor) - 带有分布式信息的分布式 Tensor。

返回
:::::::::
paddle.Tensor: 不带分布式信息的普通 Tensor，包含 ``dist_tensor`` 的全局数据。


代码示例
:::::::::

COPY-FROM: paddle.distributed.unshard_dtensor
