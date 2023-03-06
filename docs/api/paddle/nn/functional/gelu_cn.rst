.. _cn_api_nn_cn_gelu:

gelu
-------------------------------

.. py:function:: paddle.nn.functional.gelu(x, approximate=False, name=None)

gelu 激活层（GELU Activation Operator）

逐元素计算 gelu 激活函数。更多细节请参考 `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ 。

如果使用近似计算：

.. math::
    gelu(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

如果不使用近似计算：

.. math::
    gelu(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::::
 - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
 - **approximate** (bool，可选) - 是否使用近似计算，默认值为 False，表示不使用近似计算。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.gelu
