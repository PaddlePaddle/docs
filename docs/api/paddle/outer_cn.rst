.. _cn_api_tensor_outer:

outer
-------------------------------

.. py:function:: paddle.outer(x, y, name=None)


计算两个 Tensor 的外积。

对于 1 维 Tensor 正常计算外积，对于大于 1 维的 Tensor 先展平为 1 维再计算外积。

参数
:::::::::

    - **x** (Tensor) - 一个 N 维 Tensor 或者标量 Tensor。
    - **y** (Tensor) - 一个 N 维 Tensor 或者标量 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor, x、y 的外积结果，Tensor shape 为 [x.size, y.size]。

代码示例：
::::::::::

COPY-FROM: paddle.outer
