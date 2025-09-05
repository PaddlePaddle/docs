.. _cn_api_paddle_zeros:

zeros
-------------------------------

.. py:function:: paddle.zeros(shape, dtype=None, name=None, *, out=None, device=None, requires_grad=False, pin_memory=False)



创建形状为 ``shape`` 、数据类型为 ``dtype`` 且值全为 0 的 Tensor。

.. note::
    别名支持: 参数名 ``size`` 可替代 ``shape``。
    ``shape`` 支持可变参数类型。
    使用实例：
        ``paddle.zeros(1, 2, 3, dtype=paddle.float32)``
        ``paddle.zeros(size=[1, 2, 3], dtype=paddle.float32)``

参数
::::::::::::

    - **shape** (list|tuple|Tensor|*shape) - 生成的 Tensor 的形状。数据类型为 int32 或 int64。
      如果 ``shape`` 是 list、tuple，则其中的元素可以是 int，或者是形状为 [] 且数据类型为 int32、int64 的 0-D Tensor。
      如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的 1-D Tensor，表示一个列表。
      如果 ``shape`` 是 \*shape，则可以直接以可变参数的形式传入多个整数（例如 ``randn(2, 3)``）。
      该参数的别名为 ``size``。
    - **dtype** (np.dtype|str，可选) - 输出 Tensor 的数据类型，数据类型必须为 bool、float16、float32、float64、int32 或 int64。若为 None，数据类型为 float32，默认为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 用于存储结果的 Tensor。若指定，将直接写入该 Tensor，默认值为 None。
    - **device** (PlaceLike|None，可选) - 期望创建 Tensor 所在的设备。默认值为 None，表示使用当前全局设备（可通过 ``paddle.device.set_device`` 设置）。
    - **requires_grad** (bool，可选) - 是否需要为返回的 Tensor 记录梯度信息。默认值为 False。
    - **pin_memory** (bool，可选) - 若为 True，返回的 CPU Tensor 将分配在锁页内存中。仅对 CPU Tensor 生效。默认值为 False。

返回
::::::::::::
值全为 0 的 Tensor，数据类型和 ``dtype`` 定义的类型一致。


代码示例
::::::::::::

COPY-FROM: paddle.zeros
