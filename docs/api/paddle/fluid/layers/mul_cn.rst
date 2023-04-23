.. _cn_api_fluid_layers_mul:

mul
-------------------------------

.. py:function:: paddle.fluid.layers.mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None)




mul 算子
此运算是用于对输入 x 和 y 执行矩阵乘法。
公式是：

.. math::
        Out = x * y

输入 x 和 y 都可以携带 LoD（详细程度）信息。但输出仅与输入 x 共享 LoD 信息。

参数
::::::::::::

    - **x** (Variable) - 乘法运算的第一个输入 TensorTensor/LoDTensor。
    - **y** (Variable) - 乘法运算的第二个输入 TensorTensor/LoDTensor。
    - **x_num_col_dims** (int，可选) - 默认值 1，可以将具有两个以上维度的 Tensor 作为输入。如果输入 x 是具有多于两个维度的 Tensor，则输入 x 将先展平为二维矩阵。展平规则是：前 ``num_col_dims`` 将被展平成最终矩阵的第一个维度（矩阵的高度），其余的 rank(x) - num_col_dims 维度被展平成最终矩阵的第二个维度（矩阵的宽度）。结果是展平矩阵的高度等于 x 的前 ``x_num_col_dims`` 维数的乘积，展平矩阵的宽度等于 x 的最后一个 rank(x)- ``num_col_dims`` 个剩余维度的维数的乘积。例如，假设 x 是一个 5-DTensor，形状为（2,3,4,5,6），并且 ``x_num_col_dims`` 的值为 3。则扁平化后的 Tensor 具有的形即为（2x3x4,5x6）=（24,30）。
    - **y_num_col_dims** (int，可选) - 默认值 1，可以将具有两个以上维度的 Tensor 作为输入。如果输入 y 是具有多于两个维度的 Tensor，则 y 将首先展平为二维矩阵。``y_num_col_dims`` 属性确定 y 的展平方式。有关更多详细信息，请参阅 ``x_num_col_dims`` 的注释。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Variable(Tensor)乘法运算输出 Tensor。

返回类型
::::::::::::
变量(Variable)。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.mul
