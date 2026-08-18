.. _cn_api_paddle_compat_nn_functional_log_softmax:

log_softmax
-------------------------------

.. py:function:: paddle.compat.nn.functional.log_softmax(input, dim=None, dtype=None, *, out=None)

沿指定维度计算 log-softmax。

参数
:::::::::

    - **input** (Tensor) - 输入 Tensor。
    - **dim** (int|None，可选) - 计算 log-softmax 的维度。默认值：None。
    - **dtype** (DTypeLike|None，可选) - 计算前转换到的数据类型。默认值：None。

关键字参数
:::::::::::

    - **out** (Tensor|None，可选) - 保存计算结果的 Tensor。默认值：None。

返回
:::::::::

Tensor，形状与 ``input`` 相同。
