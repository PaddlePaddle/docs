.. _cn_api_nn_LeakyReLU:

LeakyReLU
-------------------------------
.. py:class:: paddle.nn.LeakyReLU(negative_slope=0.01, name=None)

LeakyReLU 激活层

.. math::

    LeakyReLU(x)=
        \left\{
        \begin{aligned}
        &x, & & if \ x >= 0 \\
        &negative\_slope * x, & & otherwise \\
        \end{aligned}
        \right. \\

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - negative_slope (float，可选) - :math:`x < 0` 时的斜率。默认值为0.01。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::

    - input：任意形状的Tensor。
    - output：和input具有相同形状的Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.LeakyReLU