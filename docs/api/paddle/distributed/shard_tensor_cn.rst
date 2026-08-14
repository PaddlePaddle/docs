.. _cn_api_paddle_distributed_shard_tensor:

shard_tensor
-------------------------------

.. py:function:: paddle.distributed.shard_tensor(data, mesh, placements, dtype=None, place=None, stop_gradient=None)

通过已知的 ``data`` 来创建一个带有分布式信息的 Tensor，Tensor 类型为 ``paddle.Tensor``。
``data`` 可以是 scalar，tuple，list，numpy\.ndarray，paddle\.Tensor。

如果 ``data`` 已经是一个 Tensor，将其转换为一个分布式 Tensor。


参数
:::::::::

    - **data** (scalar|tuple|list|ndarray|Tensor) - 初始化 Tensor 的数据，可以是 scalar，list，tuple，numpy\.ndarray，paddle\.Tensor 类型。
    - **mesh** (paddle.distributed.ProcessMesh) - 表示进程拓扑信息的 ProcessMesh 对象。
    - **placements** (list(Placement)) - 分布式 Tensor 的切分表示列表，描述 Tensor 在 mesh 上如何切分。
    - **dtype** (str|paddle.dtype|np.dtype，可选) - 创建 Tensor 的数据类型，可以是 bool、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
      默认值为 None，如果 ``data`` 为 python 浮点类型，则从 :ref:`cn_api_paddle_get_default_dtype` 获取类型，如果 ``data`` 为其他类型，则会自动推导类型。
    - **place** (CPUPlace|CUDAPinnedPlace|CUDAPlace|str，可选) - 创建 Tensor 的设备位置，可以是 CPUPlace、CUDAPinnedPlace、CUDAPlace。默认值为 None，使用全局的 place。若为字符串，可为 ``"cpu"``、``"gpu:x"`` 或 ``"gpu_pinned"``，其中 ``x`` 为 GPU 的索引。
    - **stop_gradient** (bool|None，可选) - 是否阻断 Autograd 的梯度传导。默认值为 None。若为 None，当 ``data`` 具有 ``stop_gradient`` 属性时，返回 Tensor 的 ``stop_gradient`` 与其相同；否则为 True。

返回
:::::::::
通过 ``data`` 创建的带有分布式信息的 Tensor。


代码示例
:::::::::

COPY-FROM: paddle.distributed.shard_tensor
