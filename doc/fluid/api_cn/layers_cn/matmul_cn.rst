.. _cn_api_fluid_layers_matmul:

matmul
-------------------------------

.. py:function:: paddle.fluid.layers.matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None)

``注意，如果原始张量 x 或 y 的秩为 1 且未转置，则矩阵相乘后的前置或附加维度 1 将移除。``

输入 ``x`` 和输入 ``y`` 矩阵相乘。

两个输入的形状可为任意维度，但当任一输入维度大于3时，两个输入的维度必须相等。
实际的操作取决于 ``x`` 、 ``y`` 的维度和 ``transpose_x`` 、 ``transpose_y`` 的布尔值。具体如下：

- 如果 ``transpose`` 为真，则对应 Tensor 的后两维会转置。假定 ``x`` 是一个 shape=[D] 的一维 Tensor，则 ``x`` 非转置形状为 [1, D]，转置形状为 [D, 1]。按输入要求转置之后的输入形状需满足矩阵乘法要求，即 `x_width` 与 `y_height` 相等。

- 转置后，输入的两个 Tensor 维度将为 2-D 或 n-D，将根据下列规则矩阵相乘：
  - 如果两个矩阵都是 2-D，则同普通矩阵一样进行矩阵相乘
  - 如果任意一个矩阵是 n-D，则将其视为带 batch 的二维矩阵乘法。

参数：
    - **x** (Variable)- 输入变量，类型为 Tensor 或 LoDTensor。
    - **y** (Variable)- 输入变量，类型为 Tensor 或 LoDTensor。
    - **transpose_x** (bool) - 相乘前是否转置 x。
    - **transpose_y** (bool) - 相乘前是否转置 y。
    - **alpha** (float) - 输出比例，默认为 1.0。
    - **name** (str|None) - 该层名称（可选），如果设置为空，则自动为该层命名。

返回：Variable (Tensor / LoDTensor)，矩阵相乘后的结果。

返回类型：Variable

::

    * 例 1:

    x: [B, ..., M, K], y: [B, ..., K, N]
    out: [B, ..., M, N]

    * 例 2:

    x: [B, M, K], y: [B, K, N]
    out: [B, M, N]

    * 例 3:

    x: [B, M, K], y: [K, N]
    out: [B, M, N]

    * 例 4:

    x: [M, K], y: [K, N]
    out: [M, N]

    * 例 5:

    x: [B, M, K], y: [K]
    out: [B, M]

    * 例 6:

    x: [K], y: [K]
    out: [1]

    * 例 7:

    x: [M], y: [N]
    out: [M, N]


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    # Graph Organizing
    x = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')
    y = fluid.layers.data(name='y', shape=[3, 2], dtype='float32')
    output = fluid.layers.matmul(x, y, True, True)

    # Create an executor using CPU as an example
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # Execute
    input_x = numpy.ones([2, 3]).astype(numpy.float32)
    input_y = numpy.ones([3, 2]).astype(numpy.float32)
    res, = exe.run(fluid.default_main_program(),
                   feed={'x':input_x, 'y':input_y},
                   fetch_list=[output])
    print(res)
    '''
    Output Value:
    [[2. 2. 2.]
     [2. 2. 2.]
     [2. 2. 2.]]
    '''
