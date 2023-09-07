.. _cn_api_paddle_logcumsumexp:

logcumsumexp
-------------------------------

.. py:function:: paddle.logcumsumexp(x, axis=None, dtype=None, name=None)

计算 x 的指数的前缀和的对数。

假设输入是二维矩阵，j 是 axis 维的下标，i 是另一维的下标，那么运算结果将是

.. math::

    logcumsumexp(x)_{ij} = log \sum_{i=0}^{j}exp(x_{ij})

.. note::
   结果的第一个元素和输入的第一个元素相同。

参数
:::::::::
    - **x** (Tensor) - 需要进行操作的 Tensor。
    - **axis** (int，可选) - 指明需要计算的维度。-1 代表最后一维。默认：None，将输入展开为一维变量再进行计算。
    - **dtype** (str，可选) - 输出 Tensor 的数据类型，支持 float16、float32、float64。如果指定了，那么在执行操作之前，输入 Tensor 将被转换为 dtype。这对于防止数据类型溢出非常有用。默认为：None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - Tensor (Tensor)，x 的指数的前缀和的对数。


代码示例
:::::::::

COPY-FROM: paddle.logcumsumexp
