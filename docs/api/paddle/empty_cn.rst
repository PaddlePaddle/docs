.. _cn_api_paddle_empty:

empty
-------------------------------

.. py:function:: paddle.empty(shape, dtype=None, name=None, *, out=None, device=None, requires_grad=False, pin_memory=False)



创建形状大小为 shape 并且数据类型为 dtype 的 Tensor，其中元素值是未初始化的。

.. note::
    别名支持: 参数名 ``size`` 可替代 ``shape``。
    ``shape`` 支持可变参数类型。
    使用实例：
        ``paddle.empty(1, 2, 3, dtype=paddle.float32)``
        ``paddle.empty(size=[1, 2, 3], dtype=paddle.float32)``

参数
::::::::::::

    - **shape** (list|tuple|Tensor) – 指定创建 Tensor 的形状(shape)，数据类型为 int32 或者 int64。
    - **size** - ``shape`` 的别名，行为完全一致。
    - **dtype** (np.dtype|str，可选)- 输出变量的数据类型，可以是 bool、float16、float32、float64、int32、int64、complex64、complex128。若为 None，则输出变量的数据类型为系统全局默认类型，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 用于存储结果的 Tensor。若指定，将直接写入该 Tensor，默认值为 None。
    - **device** (PlaceLike|None，可选) - 期望创建 Tensor 所在的设备。若为 None，则与 ``self`` 保持一致。
    - **requires_grad** (bool，可选) - 是否需要为返回的 Tensor 记录梯度信息。默认值为 False。
    - **pin_memory** (bool，可选) - 若为 True，返回的 CPU Tensor 将分配在锁页内存中。仅对 CPU Tensor 生效。默认值为 False。

返回
::::::::::::
返回一个根据 ``shape`` 和 ``dtype`` 创建并且尚未初始化的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.empty
