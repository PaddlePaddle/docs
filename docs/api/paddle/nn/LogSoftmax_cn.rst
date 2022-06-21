.. _cn_api_nn_LogSoftmax:

LogSoftmax
-------------------------------
.. py:class:: paddle.nn.LogSoftmax(axis=-1, name=None)

LogSoftmax激活层，计算公式如下：

.. math::

    \begin{aligned} 
    Out[i, j] &= log(softmax(x)) \\
    &= log(\frac{\exp(X[i, j])}{\sum_j(\exp(X[i, j])})
    \end{aligned}

参数
:::::::::
    - axis (int，可选) - 指定对输入Tensor进行运算的轴。``axis`` 的有效范围是[-D, D)，D是输入Tensor的维度，``axis`` 为负值时与 :math:`axis + D` 等价。默认值为-1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::

    - input：任意形状的Tensor。
    - output：和input具有相同形状的Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.LogSoftmax