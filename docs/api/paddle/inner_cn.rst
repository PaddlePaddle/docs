.. _cn_api_paddle_inner:

inner
-------------------------------

.. py:function:: paddle.inner(x, y, name=None)


计算两个 Tensor 的内积。

对于 1 维 Tensor 计算普通内积，对于大于 1 维的 Tensor 计算最后一个维度的乘积和，此时两个输入 Tensor 最后一个维度长度需要相等。

参数
::::::::::::

    - **x** (Tensor) - 一个 N 维 Tensor 或者标量 Tensor，如果是 N 维 Tensor 最后一个维度长度需要跟 y 保持一致。
    - **y** (Tensor) - 一个 N 维 Tensor 或者标量 Tensor，如果是 N 维 Tensor 最后一个维度长度需要跟 x 保持一致。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor, x、y 的内积结果，Tensor shape 为 x.shape[:-1] + y.shape[:-1]。

代码示例：
::::::::::

COPY-FROM: paddle.inner
