.. _cn_api_paddle_baddbmm:

baddbmm
-------------------------------

.. py:function:: paddle.baddbmm(input, x, y, beta=1.0, alpha=1.0, out_dtype=None, name=None, out=None)




计算 x 和 y 的批量矩阵乘积，将结果乘以标量 alpha，再加上 input 与标量 beta 的乘积，得到输出。其中 input 与 x、y 乘积的维度必须是可广播的。

计算过程的公式为：

..  math::
    out = \beta \times input + \alpha \times x \times y
其中 :math:`\beta` 和 :math:`\alpha` 是缩放因子。

..  note::
    别名支持: 参数名 ``batch1`` 可替代 ``x``，参数名 ``batch2`` 可替代 ``y`` ，如 ``baddbmm(input=tensor_input, batch1=tensor_x, batch2=tensor_y, ...)`` 等价于 ``baddbmm(input=tensor_input, x=tensor_x, y=tensor_y, ...)`` 。

参数
::::::::::::

    - **input** (Tensor) - 输入 Tensor input，必须是一个 2 维或 3 维张量，数据类型支持 bfloat16、float16、float32、float64。
    - **x** (Tensor) - 输入 Tensor x，必须是一个形状为 [b, n, p] 的 3 维张量，数据类型支持 bfloat16、float16、float32、float64。
      ``别名：batch1``
    - **y** (Tensor) - 输入 Tensor y，必须是一个形状为 [b, p, m] 的 3 维张量，数据类型支持 bfloat16、float16、float32、float64。
      ``别名：batch2``
    - **beta** (float，可选) - 乘以 input 的标量，数据类型支持 float，默认值为 1.0。
    - **alpha** (float，可选) - 乘以 x*y 的标量，数据类型支持 float，默认值为 1.0。
    - **out_dtype** (paddle.dtype, 可选) - 输出数据类型，默认值为 None，表示输出数据类型与输入 input 数据类型一致。支持设置为以下数据类型：float16、bfloat16、float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **out** (Tensor, 可选) - 用于存储输出结果的 Tensor，必须是一个形状为 [b, n, m] 的 3 维张量，默认值为 None。若指定该参数，输出结果将存储在该 Tensor 中。

返回
::::::::::::
计算得到的 Tensor。Tensor 数据类型与输入 input 数据类型一致。

代码示例
::::::::::::

COPY-FROM: paddle.baddbmm
