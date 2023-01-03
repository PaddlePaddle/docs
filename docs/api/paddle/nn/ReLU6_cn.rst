.. _cn_api_nn_ReLU6:

ReLU6
-------------------------------
.. py:class:: paddle.nn.ReLU6(name=None)

ReLU6 激活层

.. math::

    ReLU6(x) = min(max(0,x), 6)

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状：
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.ReLU6

输出 (x)
:::::::::
    定义每次调用时执行的计算。应被所有子类覆盖。

参数
:::::::::
    - **inputs** (tuple) - 未压缩的 tuple 参数。
    - **kwargs** (dict) - 未压缩的字典参数。

extra_repr()
:::::::::
    该层为额外层，您可以自定义实现层。
