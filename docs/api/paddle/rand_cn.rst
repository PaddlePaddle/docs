.. _cn_api_paddle_rand:

rand
----------------------

.. py:function:: paddle.rand(shape, dtype=None, name=None, *, out=None, device=None, pin_memory=False, requires_grad=False)

返回符合均匀分布的、范围在[0, 1)的 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

参数
::::::::::
    - **shape** (list|tuple|Tensor) - 生成的随机 Tensor 的形状。如果 ``shape`` 是 list、tuple，则其中的元素可以是 int，或者是形状为[]且数据类型为 int32、int64 的 0-D Tensor。如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的 1-D Tensor。参数名别名为 ``size``，例如 ``rand(size=[2, 3])`` 等价于 ``rand(shape=[2, 3])``；也可直接以可变长度整数参数传入形状，如 ``rand(2, 3)``。
    - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Tensor 的数据类型，支持 float32、float64。默认值为 None，此时使用全局默认数据类型（详细信息请见 :ref:`cn_api_paddle_get_default_dtype` ）。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::
    - **out** (Tensor，可选) - 输出 Tensor。
    - **device** (PlaceLike|None，可选) - 返回 Tensor 的期望设备。
    - **pin_memory** (bool，可选) - 若为 True，返回 Tensor 分配在锁页内存中。仅对 CPU Tensor 生效。默认值为 False。
    - **requires_grad** (bool，可选) - 是否由自动微分记录返回 Tensor 上的操作。默认值为 False。

返回
::::::::::
    Tensor：符合均匀分布的范围为[0, 1)的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

示例代码
::::::::::

COPY-FROM: paddle.rand
