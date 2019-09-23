.. _cn_api_fluid_layers_isfinite:

isfinite
-------------------------------

.. py:function:: paddle.fluid.layers.isfinite(x)

``注意：此算子的输入 Tensor / LoDTensor 数据类型必须为 int32 / float / double 之一。``

测试 x 是否包含无穷值（即 nan 或 inf）。若元素均为有穷数，返回真；否则返回假。

参数：
  - **x(variable)** - 变量。变量中包含被测试的 Tensor / LoDTensor

返回: Variable (Tensor)。此 Tensor 变量包含一个 bool 型结果

返回类型：Variable (Tensor)，一个包含 Tensor 的变量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    # Graph Organizing
    var = fluid.layers.data(name="data", shape=(4, 6), dtype="float32")
    output = fluid.layers.isfinite(var)

    # Create an executor using CPU as an example
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # Execute
    img = numpy.array((6, 7, 8)).astype(numpy.float32)
    res, = exe.run(fluid.default_main_program(), feed={'data':img}, fetch_list=[output])
    print(res)  # Output Value: [ True]




