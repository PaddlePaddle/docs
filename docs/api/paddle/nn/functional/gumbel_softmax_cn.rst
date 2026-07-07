.. _cn_api_paddle_nn_functional_gumbel_softmax:

gumbel_softmax
-------------------------------
.. py:function:: paddle.nn.functional.gumbel_softmax(x, temperature = 1.0, hard = False, axis = -1, name = None)

该算子实现了从 Gumbel-Softmax 分布中采样，通过 hard 可选择是否离散化。记 temperature 为 ``t``，计算过程如下：

1. 产生 gumbel 噪声

.. math::

    G_i = -log(-log(U_i)),\ U_i \sim U(0,1)

2. 对输入 ``x`` 添加噪声

.. math::

    v = [x_1 + G_1,...,x_n + G_n]

3. 计算 gumbel_softmax 并生成样本

.. math::

    gumbel\_softmax(v_i)=\frac{e^{v_i/t}}{\sum_{j=1}^n{e^{v_j/t}}},i=1,2,3...n

.. note::
    此 API 有两种调用格式：
    1. ``paddle.nn.functional.gumbel_softmax(x, temperature=1.0, hard=False, axis=-1, name=None)`` (Paddle 风格)：标准 Paddle API 签名。
    2. ``paddle.nn.functional.gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1)`` (PyTorch 风格)：兼容 PyTorch 的签名，其中 ``logits`` 是 ``x`` 的别名，``tau`` 是 ``temperature`` 的别名，``dim`` 是 ``axis`` 的别名，``eps`` 参数被接受但不产生任何作用（已废弃）。


参数
::::::::::
    - **x** (Tensor) - 一个 N-D Tensor，前 N-1 维用于独立分布 batch 的索引，最后一维表示每个类别的概率向量，dtype 类型为 float16、float32、float64。别名 ``logits``。
    - **temperature** (float，可选) - 非负标量温度值。默认值：1.0。别名 ``tau``。
    - **hard** (bool，可选) - 如果为 True，返回的样本将被离散化为 one-hot 向量，但在自动求导时仍按软样本进行微分。如果为 False，返回软样本。默认值：False。
    - **axis** (int，可选) - 沿着该维度计算 softmax。默认值：-1。别名 ``dim``。
    - **name** (str|None，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    与 ``x`` 形状相同的从 Gumbel-Softmax 分布采样的 ``Tensor``。如果 ``hard=True``，则返回的样本将是 one-hot 向量；如果 ``hard=False``，则返回的向量将是沿 ``axis`` 维度之和为 1 的概率分布。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.gumbel_softmax
