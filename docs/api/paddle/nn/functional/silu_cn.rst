.. _cn_api_nn_cn_silu:

silu
-------------------------------

.. py:function:: paddle.nn.functional.silu(x, name=None)

silu 激活层。计算公式如下：

.. math::

    silu(x) = \frac{x}{1 + \mathrm{e}^{-x}}

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::

    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为 float32 或 float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::

    ``Tensor``，数据类型和形状同 :attr:`x` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.silu
