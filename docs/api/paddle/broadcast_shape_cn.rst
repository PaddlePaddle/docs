.. _cn_api_tensor_broadcast_shape:

broadcast_shape
-------------------------------

.. py:function:: paddle.broadcast_shape(x_shape, y_shape)

.. note::
    该函数返回对 x_shape 大小的 Tensor 和 y_shape 大小的 Tensor 做 broadcast 操作后得到的 shape，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::
    - **x_shape** (list[int]|tuple[int]) - 输入 Tensor 的 shape。
    - **y_shape** (list[int]|tuple[int]) - 输入 Tensor 的 shape。

返回
:::::::::
broadcast 操作后的 shape，返回类型为 list[int]。


代码示例
:::::::::

COPY-FROM: paddle.broadcast_shape
