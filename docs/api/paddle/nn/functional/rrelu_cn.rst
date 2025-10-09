.. _cn_api_paddle_nn_functional_rrelu:

rrelu
-------------------------------

.. py:function:: paddle.nn.functional.rrelu(x, lower=1. / 8., upper=1. / 3., training=True, name=None)

rrelu 激活函数，应用随机纠正线性单元对神经元激活，参考论文：
`Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/abs/1505.00853>`_ 。

训练阶段对负斜率进行均匀分布随机采样：

.. math::

        rrelu(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    a * x, & & otherwise \\
                \end{array}
            \right.

其中，:math:`x` 为输入的 Tensor，:math:`a` 是服从（:math:`lower`，:math:`upper` ）均匀分布的随机值。

测试阶段负斜率取均匀分布上下边界（:math:`lower` 及 :math:`upper` ）的平均值：

.. math::

        rrelu(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    (lower + upper) * 0.5 * x, & & otherwise \\
                \end{array}
            \right.

其中，:math:`x` 为输入的 Tensor，:math:`lower` 及 :math:`upper` 是随机均匀分布的上下边界。

参数
::::::::::
    - **x** (Tensor) - 输入的 `Tensor`，数据类型为：float16、float32、float64。
    - **lower** (float，可选) - 负值斜率的随机值范围下限，`lower` 包含在范围中。支持的数据类型：float。默认值为 0.125。
    - **upper** (float，可选) - 负值斜率的随机值范围上限，`upper` 包含在范围中。支持的数据类型：float。默认值为 0.333。
    - **training** (bool，可选) - 标记是否为训练阶段。默认：True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.rrelu
