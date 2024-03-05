.. _cn_api_paddle_nn_functional_celu:

celu
-------------------------------

.. py:function:: paddle.nn.functional.celu(x, alpha=1.0, name=None)

celu 激活层（CELU Activation Operator）

根据 `Continuously Differentiable Exponential Linear Units <https://arxiv.org/abs/1704.07483>`_ 对输入 Tensor 中每个元素应用以下计算。

.. math::

    celu(x) = max(0, x) + min(0, \alpha * (e^{x/\alpha} − 1))

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::::

::::::::::
 - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float16、float32、float64。
 - **alpha** (float，可选) - celu 的 alpha 值，默认值为 1.0。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.celu
