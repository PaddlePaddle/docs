.. _cn_api_fluid_layers_matmul:



matmul
-------------------------------

.. py:function:: paddle.fluid.layers.matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None)

对两个张量进行矩阵相乘

当前输入的张量可以为任意阶，但当任意一个输入的阶数大于3时，两个输入的阶必须相等。
实际的操作取决于x,y的维度和 ``transpose_x`` , ``transpose_y`` 的标记值。具体如下：

- 如果transpose值为真，则对应 ``tensor`` 的最后两维将被转置。如：x是一个shape=[D]的一阶张量，那么x在非转置形式中为[1,D]，在转置形式中为[D,1],而y则相反，在非转置形式中作为[D,1]，在转置形式中作为[1,D]。

- 转置后，这两个`tensors`将为 2-D 或 n-D ,并依据下列规则进行矩阵相乘：
  - 如果两个都是2-D，则同普通矩阵一样进行矩阵相乘
  - 如果任意一个是n-D，则将其视为驻留在最后两个维度的矩阵堆栈，并在两个张量上应用支持广播的批处理矩阵乘法。

**注意，如果原始张量x或y的秩为1且没有转置，则在矩阵乘法之后，前置或附加维度1将被移除。**


参数：
    - **x** (Variable)-输入变量，类型为Tensor或LoDTensor
    - **y** (Variable)-输入变量，类型为Tensor或LoDTensor
    - **transpose_x** (bool)-相乘前是否转置x
    - **transpose_y** (bool)-相乘前是否转置y
    - **alpha** (float)-输出比例。默认为1.0
    - **name** (str|None)-该层名称（可选）。如果设置为空，则自动为该层命名

返回：张量乘积变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    # 以下是解释输入和输出维度的示例
    # x: [B, ..., M, K], y: [B, ..., K, N]
    # fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

    # x: [B, M, K], y: [B, K, N]
    # fluid.layers.matmul(x, y)  # out: [B, M, N]

    # x: [B, M, K], y: [K, N]
    # fluid.layers.matmul(x, y)  # out: [B, M, N]

    # x: [M, K], y: [K, N]
    # fluid.layers.matmul(x, y)  # out: [M, N]

    # x: [B, M, K], y: [K]
    # fluid.layers.matmul(x, y)  # out: [B, M]

    # x: [K], y: [K]
    # fluid.layers.matmul(x, y)  # out: [1]

    # x: [M], y: [N]
    # fluid.layers.matmul(x, y, True, True)  # out: [M, N]

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')
    y = fluid.layers.data(name='y', shape=[3, 2], dtype='float32')
    out = fluid.layers.matmul(x, y, True, True)







