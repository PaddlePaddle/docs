.. _cn_api_nn_functional_normalize:

normalize
-------------------------------

.. py:function:: paddle.nn.functional.normalize(x, p=2, axis=1, epsilon=1e-12, name=None)

使用 :math:`L_p` 范数沿维度 ``axis`` 对 ``x`` 进行归一化。计算公式如下：

.. math::

    y = \frac{x}{ \max\left( \lvert \lvert x \rvert \rvert_p, epsilon\right) }

.. math::
    \lvert \lvert x \rvert \rvert_p = \left(\sum_i {\lvert x_i\rvert^p}  \right)^{1/p}

其中 :math:`\sum_i{\lvert x_i\rvert^p}` 沿维度 ``axis`` 进行计算。


参数
:::::::::
    - **x** (Tensor) - 输入可以是 N-D Tensor。数据类型为：float32、float64。
    - **p** (float|int，可选) - 范数公式中的指数值。默认值：2
    - **axis** (int，可选）- 要进行归一化的轴。如果 ``x`` 是 1-D Tensor，轴固定为 0。如果 `axis < 0`，轴为 `x.ndim + axis`。-1 表示最后一维。
    - **epsilon** (float，可选) - 添加到分母上的值以防止分母为 0。默认值为 1e-12。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，输出的形状和数据类型和 ``x`` 相同。


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.normalize
