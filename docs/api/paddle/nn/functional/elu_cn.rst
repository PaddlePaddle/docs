.. _cn_api_nn_cn_elu:

elu
-------------------------------

.. py:function:: paddle.nn.functional.elu(x, alpha=1.0, name=None)

elu 激活层（ELU Activation Operator）

根据 `Exponential Linear Units <https://arxiv.org/abs/1511.07289>`_ 对输入 Tensor 中每个元素应用以下计算。

.. math::

    elu(x)=
        \left\{
            \begin{array}{lcl}
            x,& &\text{if } \ x > 0 \\
            alpha * (e^{x} - 1),& &\text{if } \ x <= 0
            \end{array}
        \right.

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::::

::::::::::
 - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
 - **alpha** (float，可选) - elu 的 alpha 值，默认值为 1.0。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.elu
