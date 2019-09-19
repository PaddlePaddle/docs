.. _cn_api_fluid_layers_hard_swish:

hard_swish
-------------------------------

.. py:function:: paddle.fluid.layers.hard_swish(x, threshold=6.0, scale=6.0, offset=3.0, name=None)

hard_swish激活函数，swish的hard version(https://arxiv.org/pdf/1905.02244.pdf)。

 :math:`out = \frac{x * (min(max(0, x+offset), threshold))}{scale}`

 `` threshold`` 和 ``scale`` 应该为正， ``offset`` 正负均可，默认参数如上，建议使用默认参数。

参数：
    - **x** (Variable) - 要做激活操作的输入变量。
    - **threshold** (float) - 做激活操作的threshold，默认为6.0。
    - **scale** (float) - 激活操作的scale，默认为6.0。
    - **offset** (float) - 激活操作的offset，默认为3.0。
    - **name** (str|None) - 层名，设置为None，此层将被自动命名。
    
返回：与输入有着同样shape的输出变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.hard_swish(x)






