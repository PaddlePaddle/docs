.. _cn_api_paddle_nn_GELU:

GELU
-------------------------------
.. py:class:: paddle.nn.GELU(approximate=False, name=None)

GELU 激活层（GELU Activation Operator）

逐元素计算 GELU 激活函数。更多细节请参考 `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ 。

如果使用近似计算：

.. math::
    GELU(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

如果不使用近似计算：

.. math::
    GELU(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))


其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **approximate** (bool，可选) - 是否使用近似计算，默认值为 False，即不使用近似计算。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.GELU
