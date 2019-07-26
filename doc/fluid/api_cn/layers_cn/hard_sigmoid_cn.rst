.. _cn_api_fluid_layers_hard_sigmoid:

hard_sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.hard_sigmoid(x, slope=0.2, offset=0.5, name=None)

HardSigmoid激活算子。

sigmoid的分段线性逼近(https://arxiv.org/abs/1603.00391)，比sigmoid快得多。

.. math::

      \\out=\max(0,\min(1,slope∗x+shift))\\

斜率是正数。偏移量可正可负的。斜率和位移的默认值是根据上面的参考设置的。建议使用默认值。

参数：
    - **x** (Variable) - HardSigmoid operator的输入
    - **slope** (FLOAT|0.2) -斜率
    - **offset** (FLOAT|0.5)  - 偏移量
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。


**代码示例：**


.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.hard_sigmoid(x, slope=0.3, offset=0.8)






