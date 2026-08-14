.. _cn_api_paddle_standard_normal:

standard_normal
-------------------------------

.. py:function:: paddle.standard_normal(shape, dtype=None, name=None, *, out=None, device=None, requires_grad=False)

返回符合标准正态分布（均值为 0，标准差为 1 的正态随机分布）的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

参数
::::::::::
  - **shape** (list|tuple|Tensor) - 生成的随机 Tensor 的形状。如果 ``shape`` 是 list、tuple，则其中的元素可以是 int，或者是形状为[]且数据类型为 int32、int64 的 0-D Tensor。如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的 1-D Tensor。
  - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Tensor 的数据类型，支持 float16、bfloat16、float32、float64、complex64、complex128。默认值为 None，此时使用全局默认数据类型（详细信息请见 :ref:`cn_api_paddle_get_default_dtype` ）。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::
  - **out** (Tensor，可选) - 输出 Tensor。
  - **device** (PlaceLike|None，可选) - 返回 Tensor 的期望设备。为 None 时，使用当前默认 Tensor 类型的当前设备。默认值为 None。
  - **requires_grad** (bool，可选) - 是否由自动微分记录返回 Tensor 上的操作。默认值为 False。

返回
::::::::::
  Tensor：符合标准正态分布的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

示例代码
::::::::::

COPY-FROM: paddle.standard_normal
