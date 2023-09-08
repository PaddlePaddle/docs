.. _cn_api_paddle_nn_RReLU:

RReLU
-------------------------------
.. py:class:: paddle.nn.RReLU(lower=0.125, upper=0.3333333333333333, name=None)

RReLU 激活层，应用随机纠正线性单元对神经元激活，参考论文：
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
    - **lower** (float，可选) - 负值斜率的随机值范围下限，`lower` 包含在范围中。支持的数据类型：float。默认值为 0.125。
    - **upper** (float，可选) - 负值斜率的随机值范围上限，`upper` 包含在范围中。支持的数据类型：float。默认值为 0.3333333333333333。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - **x** (Tensor) – 任意形状的 Tensor，默认数据类型为 float32。
    - **out** (Tensor) – 和 x 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.RReLU
