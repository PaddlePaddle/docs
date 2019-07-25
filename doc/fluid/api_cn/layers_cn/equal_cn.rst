.. _cn_api_fluid_layers_equal:

equal
-------------------------------

.. py:function:: paddle.fluid.layers.equal(x,y,cond=None)

**equal**
该层返回 :math:`x==y` 按逐元素运算而得的真值。

参数：
    - **x** (Variable)-equal的第一个操作数
    - **y** (Variable)-equal的第二个操作数
    - **cond** (Variable|None)-输出变量（可选），用来存储equal的结果

返回：张量类型的变量，存储equal的输出结果

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[3,10,32,32], dtype="float32")
    limit = fluid.layers.data(name="limit", shape=[3,10,32,32], dtype="float32")
    less = fluid.layers.equal(x=label,y=limit)




