.. _cn_api_paddle_randint:

randint
-------------------------------

.. py:function:: paddle.randint(low=0, high=None, shape=[1], dtype=None, name=None, *, out=None, device=None, pin_memory=False, requires_grad=False)

返回服从均匀分布的、范围在[``low``, ``high``)的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。当 ``high`` 为 None 时（默认），均匀采样的区间为[0, ``low``)。

参数
::::::::::
    - **low** (int，可选) - 要生成的随机值范围的下限，``low`` 包含在范围中。当 ``high`` 为 None 时，均匀采样的区间为[0, ``low``)。默认值为 0。
    - **high** (int，可选) - 要生成的随机值范围的上限，``high`` 不包含在范围中。默认值为 None，此时范围是[0, ``low``)。
    - **shape** (list|tuple|Tensor，可选) - 生成的随机 Tensor 的形状。如果 ``shape`` 是 list、tuple，则其中的元素可以是 int，或者是形状为[]且数据类型为 int32、int64 的 0-D Tensor。如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的 1-D Tensor。默认值为[1]。别名 ``size``。
    - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Tensor 的数据类型，支持 int32、int64。当该参数值为 None 时， 输出 Tensor 的数据类型为 int64。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。
    - **device** (PlaceLike，可选) - 返回 Tensor 所在的设备。默认值为 ``None``。
    - **pin_memory** (bool，可选) - 如果设置为 True，返回的 Tensor 将分配在锁页内存中。默认值为 False。
    - **requires_grad** (bool，可选) - 是否需要自动求导记录返回 Tensor 上的操作。默认值为 False。

返回
::::::::::
    服从均匀分布的、范围在[``low``, ``high``)的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

代码示例
:::::::::::

COPY-FROM: paddle.randint
