.. _cn_api_fluid_layers_one_hot:

one_hot
-------------------------------

.. py:function:: paddle.fluid.layers.one_hot(input, depth)

该层创建输入指数的one-hot表示

参数：
    - **input** (Variable)-输入指数，最后维度必须为1
    - **depth** (scalar)-整数，定义one-hot维度的深度

返回：输入的one-hot表示

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    one_hot_label = fluid.layers.one_hot(input=label, depth=10)









