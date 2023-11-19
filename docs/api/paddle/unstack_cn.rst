.. _cn_api_paddle_unstack:

unstack
-------------------------------

.. py:function:: paddle.unstack(x, axis=0, num=None)




将单个 dim 为 ``D`` 的 Tensor 沿 ``axis`` 轴 unpack 为 ``num`` 个 dim 为 ``(D-1)`` 的 Tensor。

参数
::::::::::::

      - **x** (Tensor) – 输入 x 为 ``dim > 0`` 的 Tensor，
      支持的数据类型：float32，float64，int32，int64， complex64，complex128。

      - **axis** (int | 可选) – 输入 Tensor 进行 unpack 运算所在的轴，axis 的范围为：``[-D, D)`` ，
      如果 ``axis < 0``，则 :math:`axis = axis + dim(x)`，axis 的默认值为 0。

      - **num** (int | 可选) - axis 轴的长度，一般无需设置，默认值为 ``None`` 。

返回
::::::::::::
 长度为 num 的 Tensor 列表，数据类型与输入 Tensor 相同，dim 为 ``(D-1)``。


代码示例
::::::::::::

COPY-FROM: paddle.unstack
