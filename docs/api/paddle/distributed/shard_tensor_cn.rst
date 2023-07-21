.. _cn_api_distributed_shard_tensor:

shard_tensor
-------------------------------

.. py:function:: paddle.shard_tensor(data, dtype=None, place=None, stop_gradient=True, dist_attr=None)

通过已知的 ``data`` 来创建一个带有分布式信息的 Tensor，Tensor 类型为 ``paddle.Tensor``。
``data`` 可以是 scalar，tuple，list，numpy\.ndarray，paddle\.Tensor。

如果 ``data`` 已经是一个 Tensor，将其转换为一个分布式 Tensor。


参数
:::::::::

    - **data** (scalar|tuple|list|ndarray|Tensor) - 初始化 Tensor 的数据，可以是 scalar，list，tuple，numpy\.ndarray，paddle\.Tensor 类型。
    - **dtype** (str，可选) - 创建 Tensor 的数据类型，可以是 bool、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
      默认值为 None，如果 ``data`` 为 python 浮点类型，则从 :ref:`cn_api_paddle_framework_get_default_dtype` 获取类型，如果 ``data`` 为其他类型，则会自动推导类型。
    - **place** (CPUPlace|CUDAPinnedPlace|CUDAPlace，可选) - 创建 tensor 的设备位置，可以是 CPUPlace、CUDAPinnedPlace、CUDAPlace。默认值为 None，使用全局的 place。
    - **stop_gradient** (bool，可选) - 是否阻断 Autograd 的梯度传导。默认值为 True，此时不进行梯度传传导。
    - **dist_attr** (paddle.distributed.DistAttr) - 描述 Tensor 在 ProcessMesh 上的分布或切片方式。

返回
:::::::::
通过 ``data`` 创建的带有分布式信息的 Tensor。


代码示例
:::::::::

COPY-FROM: paddle.distributed.shard_tensor
