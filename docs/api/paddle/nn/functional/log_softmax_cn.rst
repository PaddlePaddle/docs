.. _cn_api_paddle_nn_functional_log_softmax:

log_softmax
-------------------------------
.. py:function:: paddle.nn.functional.log_softmax(x, axis=-1, dtype=None, name=None)

实现了 log_softmax 层。计算公式如下：

.. math::

    \begin{aligned}
    log\_softmax[i, j] &= log(softmax(x)) \\
    &= log(\frac{\exp(X[i, j])}{\sum_j(\exp(X[i, j])})
    \end{aligned}

参数
::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
    - **axis** (int，可选) - 指定对输入 ``x`` 进行运算的轴。``axis`` 的有效范围是[-D, D)，D 是输入 ``x`` 的维度，``axis`` 为负值时与 :math:`axis + D` 等价。默认值为-1。
    - **dtype** (str|np.dtype|core.VarDesc.VarType，可选) - 输入 Tensor 的数据类型。如果指定了 ``dtype``，则输入 Tensor 的数据类型会在计算前转换到 ``dtype`` 。 ``dtype``  可以用来避免数据溢出。如果 ``dtype`` 为 None，则输出 Tensor 的数据类型和 ``x`` 相同。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，形状和 ``x`` 相同，数据类型为 ``dtype`` 或者和 ``x`` 相同。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.log_softmax
