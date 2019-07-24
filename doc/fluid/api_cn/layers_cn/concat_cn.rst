.. _cn_api_fluid_layers_concat:

concat
-------------------------------

.. py:function:: paddle.fluid.layers.concat(input,axis=0,name=None)

**Concat**

这个函数将输入连接在前面提到的轴上，并将其作为输出返回。

参数：
    - **input** (list)-将要联结的张量列表
    - **axis** (int)-数据类型为整型的轴，其上的张量将被联结
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名。

返回：输出的联结变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    
    import paddle.fluid as fluid
    a = fluid.layers.data(name='a', shape=[2, 13], dtype='float32')
    b = fluid.layers.data(name='b', shape=[2, 3], dtype='float32')
    c = fluid.layers.data(name='c', shape=[2, 2], dtype='float32')
    d = fluid.layers.data(name='d', shape=[2, 5], dtype='float32')
    out = fluid.layers.concat(input=[Efirst, Esecond, Ethird, Efourth])









