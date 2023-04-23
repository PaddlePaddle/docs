.. _cn_api_nn_cn_gumbel_softmax:

gumbel_softmax
-------------------------------
.. py:function:: paddle.nn.functional.gumbel_softmax(x, temperature = 1.0, hard = False, axis = -1, name = None)

实现了按 Gumbel-Softmax 分布进行采样的功能，通过 hard 可选择是否离散化。记 temperature 为 ``t``，涉及到的等式如下：

1. 产生 gumbel 噪声

.. math::

    G_i = -log(-log(U_i)),\ U_i \sim U(0,1)

2. 对输入 ``x`` 添加噪声

.. math::

    v = [x_1 + G_1,...,x_n + G_n]

3. 计算 gumbel_softmax

.. math::

    gumbel\_softmax(v_i)=\frac{e^{v_i/t}}{\sum_{j=1}^n{e^{v_j/t}}},i=1,2,3...n


参数
::::::::::
    - **x** (Tensor) - 一个 N-D Tensor，前 N-1 维用于独立分布 batch 的索引，最后一维表示每个类别的概率，dtype 类型为 float16，float，double。
    - **temperature** (float，可选) - 大于 0 的标量。默认值：1.0。
    - **hard** (bool，可选) - 如果是 True，返回离散的 one-hot 向量。如果是 False，返回软样本。默认值：False。
    - **axis** (int，可选) - 按照维度 axis 计算 softmax。默认值：-1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    与 ``x`` 形状相同的符合 gumbel-softmax 分布的 ``Tensor``。如果 ``hard=True``，则返回的样本将是 one-hot。如果 ``hard=False``，则返回的向量将是各维度加起来等于 1 的概率。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.gumbel_softmax
