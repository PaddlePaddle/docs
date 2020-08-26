.. _cn_api_fluid_layers_ones_like:

ones_like
-------------------------------

.. py:function:: paddle.fluid.layers.ones_like(x, out=None)




ones_like

该功能创建一个形状与类型与x相似的张量，初始值为1。


参数：
    - **x** (Variable) - 指定形状与数据类型的输入张量
    - **out** (Variable)-输出张量

返回：输出张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name='x', dtype='float32', shape=[3], append_batch_size=False)
    data = fluid.layers.ones_like(x) # [1.0, 1.0, 1.0]



