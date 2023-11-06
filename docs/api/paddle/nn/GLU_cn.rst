.. _cn_api_paddle_nn_GLU:

GLU
-------------------------------
.. py:class:: paddle.nn.GLU(axis=-1, name=None)

GLU 激活层（GLU Activation Operator）

门控线性单元。输入按照给定的维度二等分，其中第一部分被用作内容，第二部分经过一个 sigmoid 函数之后被用作门限。输出是内容和门限的逐元素乘积。更多细节请参考 `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_ 。

.. math::
    GLU(a, b) = a \otimes \sigma(b)

参数
::::::::::
    - **axis** (int，可选) - 沿着该轴将输入二等分。:math:`D` 为输入的维数，则 :attr:`axis` 应该在 :math:`[-D, D)` 的范围内。如 :attr:`axis` 为负数，则相当于 :math:`axis + D`，默认值为-1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：shape[axis]为偶数的 Tensor。
    - output： 数据类型与输入一致，在指定的轴上其尺寸减半的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.GLU
