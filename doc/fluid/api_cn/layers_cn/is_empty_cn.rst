.. _cn_api_fluid_layers_is_empty:

is_empty
-------------------------------

.. py:function:: paddle.fluid.layers.is_empty(x, cond=None)

测试变量是否为空

参数：
    - **x** (Variable)-测试的变量
    - **cond** (Variable|None)-可选输出参数，默认为空（None）。若传入了该参数，则该参数中存储返回给定x的测试结果

返回：布尔类型的标量。如果变量x为空则值为真

返回类型：Variable

抛出异常：``TypeError``-如果input类型不是Variable或cond存储的返回结果的类型不是bool

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
    res = fluid.layers.is_empty(x=input)
    # or:
    # fluid.layers.is_empty(x=input, cond=res)




