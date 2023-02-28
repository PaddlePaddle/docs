.. _cn_api_nn_LogSigmoid:

LogSigmoid
-------------------------------
.. py:class:: paddle.nn.LogSigmoid(name=None)

LogSigmoid 激活层。用于创建一个 LogSigmoid 的可调用类，这个类可以计算输入 x 经过激活函数 LogSigmoid 之后的值，LogSigmoid 激活函数计算公式如下：

.. math::

    LogSigmoid(x) = \log \frac{1}{1 + e^{-x}}

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

返回
::::::::::
返回计算 LogSigmoid 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.nn.LogSigmoid
