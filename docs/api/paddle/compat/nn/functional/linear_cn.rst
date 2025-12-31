.. _cn_api_paddle_compat_nn_functional_linear:

linear
-------------------------------

.. py:function:: paddle.compat.nn.functional.linear(input, weight, bias=None)

PyTorch 兼容的线性函数版本：

    - 与 PyTorch 一致的数学意义：

    .. math::

        Out = X W^T + b

    其中 :math:`W` 是权重张量 ``weight`` ， :math:`b` 是偏置张量 ``bias``， :math:`X` 是输入张量 ``input``

    - 此实现与 PyTorch 的线性函数对齐，计算 :math:`y = XW^T + b`。


参数
:::::::::::

        - **input** (Tensor) - 输入张量。数据类型应为 bfloat16、float16、float32 或 float64。输入张量的形状应为 :math:`[*, in\_features]`，其中 :math:`*` 表示任意数量的额外维度，包括无。
        - **weight** (Tensor) - 权重张量。数据类型应为 float16、float32 或 float64。形状应为 :math:`[out\_features, in\_features]`。
        - **bias** (Tensor, 可选) - 偏置张量。数据类型应为 float16、float32 或 float64。如果为 None，则不添加偏置到输出单元。默认值 None。


形状
:::::::::

        - **输入** : 多维张量，形状为 :math:`[*, in\_features]` 。数据类型为 bfloat16、float16、float32 或 float64。
        - **输出** : 多维张量，形状为 :math:`[*, out\_features]` 。数据类型与输入相同。


代码示例
::::::::::::

COPY-FROM: paddle.compat.nn.functional.linear
