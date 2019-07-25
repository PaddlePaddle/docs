.. _cn_api_fluid_layers_argmax:

argmax
-------------------------------

.. py:function:: paddle.fluid.layers.argmax(x,axis=0)

**argmax**

该功能计算输入张量元素中最大元素的索引，张量的元素在提供的轴上。

参数：
    - **x** (Variable)-用于计算最大元素索引的输入
    - **axis** (int)-用于计算索引的轴

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    
    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
    out = fluid.layers.argmax(x=in, axis=0)
    out = fluid.layers.argmax(x=in, axis=-1)









