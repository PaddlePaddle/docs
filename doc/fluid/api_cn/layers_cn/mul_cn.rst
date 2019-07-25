.. _cn_api_fluid_layers_mul:

mul
-------------------------------

.. py:function:: paddle.fluid.layers.mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None)

mul算子
此运算是用于对输入X和Y执行矩阵乘法。
等式是：

.. math::
        Out = X * Y

输入X和Y都可以携带LoD（详细程度）信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** (Variable)- (Tensor) 乘法运算的第一个输入张量。
        - **y** (Variable)- (Tensor) 乘法运算的第二个输入张量。
        - **x_num_col_dims** （int）- 默认值1， 可以将具有两个以上维度的张量作为输入。如果输入X是具有多于两个维度的张量，则输入X将先展平为二维矩阵。展平规则是：前 ``num_col_dims`` 将被展平成最终矩阵的第一个维度（矩阵的高度），其余的 rank(X) - num_col_dims 维度被展平成最终矩阵的第二个维度（矩阵的宽度）。结果是展平矩阵的高度等于X的前 ``x_num_col_dims`` 维数的乘积，展平矩阵的宽度等于X的最后一个秩（x）- ``num_col_dims`` 个剩余维度的维数的乘积。例如，假设X是一个五维张量，形状为（2,3,4,5,6）。 则扁平化后的张量具有的形即为 （2x3x4,5x6）=（24,30）。
        - **y_num_col_dims** （int）- 默认值1， 可以将具有两个以上维度的张量作为输入。如果输入Y是具有多于两个维度的张量，则Y将首先展平为二维矩阵。 ``y_num_col_dims`` 属性确定Y的展平方式。有关更多详细信息，请参阅 ``x_num_col_dims`` 的注释。
        - **name** (basestring | None)- 输出的名称。

返回：       乘法运算输出张量（Tensor）.

返回类型：    输出(Variable)。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    dataX = fluid.layers.data(name="dataX", append_batch_size = False, shape=[2, 5], dtype="float32")
    dataY = fluid.layers.data(name="dataY", append_batch_size = False, shape=[5, 3], dtype="float32")
    output = fluid.layers.mul(dataX, dataY,
                              x_num_col_dims = 1,
                              y_num_col_dims = 1)






