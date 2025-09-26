.. _cn_api_paddle_randn:

randn
-------------------------------

.. py:function:: paddle.randn(shape, dtype=None, name=None, *, out=None, device=None, requires_grad=False, pin_memory=False)

返回符合标准正态分布（均值为 0，标准差为 1 的正态随机分布）的随机 Tensor，形状为  ``shape`` ，数据类型为  ``dtype`` 。

参数
::::::::::

  - **shape** (list|tuple|Tensor|*shape) - 生成的 Tensor 的形状。数据类型为 int32 或 int64。
    如果  ``shape``  是 list、tuple，则其中的元素可以是 int，或者是形状为 [] 且数据类型为 int32、int64 的 0-D Tensor。
    如果  ``shape``  是 Tensor，则是数据类型为 int32、int64 的 1-D Tensor，表示一个列表。
    如果  ``shape``  是 \*shape，则可以直接以可变参数的形式传入多个整数（例如  ``randn(2, 3)`` ）。
    该参数的别名为  ``size`` 。
  - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Tensor 的数据类型。
    支持的数据类型包括：float16、bfloat16、float32、float64、complex64、complex128。
    默认值为 None，此时使用全局默认数据类型（可参考  ``get_default_dtype``  了解详情）。
  - **name** (str，可选) - 操作的名称，具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
  - **out** (Tensor，可选) - 用于保存输出结果的 Tensor。默认值为 None。
  - **device** (PlaceLike|None，可选) - 指定返回 Tensor 所在的设备。默认值为 None，表示使用当前全局设备（可通过  ``paddle.device.set_device``  设置）。
  - **requires_grad** (bool，可选) - 是否在返回的 Tensor 上记录 autograd 的操作。默认值为 False。
  - **pin_memory** (bool，可选) - 如果设置为 True，返回的 Tensor 会分配在锁页内存中。仅对 CPU Tensor 生效。默认值为 False。

返回
::::::::::
  Tensor：符合标准正态分布的随机 Tensor，形状为  ``shape`` ，数据类型为  ``dtype`` 。

示例代码
::::::::::

COPY-FROM: paddle.randn
