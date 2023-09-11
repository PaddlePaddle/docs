.. _cn_api_paddle_logsumexp:

logsumexp
-------------------------------

.. py:function:: paddle.logsumexp(x, axis=None, keepdim=False, name=None)

沿着参数 ``axis`` 计算 ``x`` 的以 e 为底的指数的和的自然对数。计算公式如下：

.. math::
   logsumexp(x) = \log\sum exp(x)

参数
::::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：float16，float32、float64，维度不超过 4 。
    - **axis** (int|list|tuple，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是 int、list(int)、tuple(int)。如果 ``axis`` 包含多个维度，则沿着 ``axis`` 中的所有轴进行计算。``axis`` 或者其中的元素值应该在范围[-D, D)内，D 是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于 0，则等价于 :math:`axis + D`。如果 ``axis`` 是 None，则对 ``x`` 的全部元素计算 logsumexp。默认值为 None。
    - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留减小的维度。如果 ``keepdim`` 为 True，则输出 Tensor 和 ``x`` 具有相同的维度(减少的维度除外，减少的维度的大小为 1)。否则，输出 Tensor 的形状会在 ``axis`` 上进行 squeeze 操作。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，沿着 ``axis`` 进行 logsumexp 计算的结果，数据类型和 ``x`` 相同。

代码示例
::::::::::

COPY-FROM: paddle.logsumexp
