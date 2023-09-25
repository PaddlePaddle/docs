.. _cn_api_paddle_nn_ThresholdedReLU:

ThresholdedReLU
-------------------------------
.. py:class:: paddle.nn.ThresholdedReLU(threshold=1.0, name=None)

Thresholded ReLU 激活层

.. math::

    ThresholdedReLU(x) = \begin{cases}
                          x, \text{if } x > threshold \\
                          0, \text{otherwise}
                         \end{cases}

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **threshold** (float，可选) - ThresholdedReLU 激活计算公式中的 threshold 值。默认值为 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.ThresholdedReLU
