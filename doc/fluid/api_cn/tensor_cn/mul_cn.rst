.. _cn_api_tensor_argmax:

mul
-------------------------------

.. py:function:: paddle.mul(x, y, x_num_col_dims=1, y_num_col_dims=1, out=None, name=None)


mul算子
此运算是用于对输入x和y执行矩阵乘法。
公式是：

.. math::
        out = x * y

输入x和y都可以携带LoD（详细程度）信息。但输出仅与输入x共享LoD信息。

参数：
    - **x** (Variable) - 乘法运算的第一个输入张量Tensor/LoDTensor。
    - **y** (Variable) - 乘法运算的第二个输入张量Tensor/LoDTensor。
    - **x_num_col_dims** (int，可选) - 默认值1， 可以将具有两个以上维度的张量作为输入。如果输入x是具有多于两个维度的张量，则输入x将先展平为二维矩阵。展平规则是：前 ``num_col_dims`` 将被展平成最终矩阵的第一个维度（矩阵的高度），其余的 rank(x) - num_col_dims 维度被展平成最终矩阵的第二个维度（矩阵的宽度）。结果是展平矩阵的高度等于x的前 ``x_num_col_dims`` 维数的乘积，展平矩阵的宽度等于x的最后一个 rank(x)- ``num_col_dims`` 个剩余维度的维数的乘积。例如，假设x是一个5-D张量，形状为（2,3,4,5,6），并且 ``x_num_col_dims`` 的值为3。 则扁平化后的张量具有的形即为（2x3x4,5x6）=（24,30）。
    - **y_num_col_dims** (int，可选) - 默认值1， 可以将具有两个以上维度的张量作为输入。如果输入y是具有多于两个维度的张量，则y将首先展平为二维矩阵。 ``y_num_col_dims`` 属性确定y的展平方式。有关更多详细信息，请参阅 ``x_num_col_dims`` 的注释。
    - **out** (Variable, 可选) - 默认值None，如果out不为空，则矩阵乘法运算结果存储在out变量中。 
    - **name** (str，可选) - 默认值None，输出的名称。该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_name`。当out和name同时不为空时，结果输出变量名与out保持一致。

返回：Variable(Tensor)乘法运算输出张量。

返回类型：变量(Variable)。

**代码示例**

.. code-block:: python
    
    import paddle
    import paddle.fluid as fluid
    dataX = fluid.data(name="dataX", shape=[2, 5], dtype="float32")
    dataY = fluid.data(name="dataY", shape=[5, 3], dtype="float32")

    res = fluid.data(name="output", shape=[2, 3], dtype="float32")
    output = paddle.mul(dataX, dataY,
                              x_num_col_dims = 1,
                              y_num_col_dims = 1, 
                              out=res)


