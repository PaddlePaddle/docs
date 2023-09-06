.. _cn_api_nn_CELU:

CELU
-------------------------------
.. py:class:: paddle.nn.CELU(alpha=1.0, name=None)

CELU 激活层（CELU Activation Operator）

根据 `Continuously Differentiable Exponential Linear Units <https://arxiv.org/abs/1704.07483>`_ 对输入 Tensor 中每个元素应用以下计算。

.. math::

    CELU(x) = max(0, x) + min(0, \alpha * (e^{x/\alpha} − 1))

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **alpha** (float，可选) - CELU 的 alpha 值，默认值为 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.CELU
