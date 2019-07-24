.. _cn_api_fluid_layers_assign:

assign
-------------------------------

.. py:function:: paddle.fluid.layers.assign(input,output=None)

该函数将输入变量复制到输出变量

参数：
    - **input** (Variable|numpy.ndarray)-源变量
    - **output** (Variable|None)-目标变量

返回：作为输出的目标变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
    out = fluid.layers.create_tensor(dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    fluid.layers.assign(hidden, out)









