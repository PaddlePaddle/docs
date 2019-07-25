.. _cn_api_fluid_layers_is_empty:

is_empty
-------------------------------

.. py:function:: paddle.fluid.layers.is_empty(x, cond=None)

测试变量是否为空

参数：
    - **x** (Variable)-测试的变量
    - **cond** (Variable|None)-输出参数。返回给定x的测试结果，默认为空（None）

返回：布尔类型的标量。如果变量x为空则值为真

返回类型：变量（Variable）

抛出异常：``TypeError``-如果input不是变量或cond类型不是变量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
    res = fluid.layers.is_empty(x=input)
    # or:
    # fluid.layers.is_empty(x=input, cond=res)




