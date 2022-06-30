.. _cn_api_nn_LogSigmoid:

LogSigmoid
-------------------------------
.. py:class:: paddle.nn.LogSigmoid(name=None)

LogSigmoid激活层。计算公式如下：

.. math::

    LogSigmoid(x) = \log \frac{1}{1 + e^{-x}}

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状：
::::::::::
    - input：任意形状的Tensor。
    - output：和input具有相同形状的Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.LogSigmoid