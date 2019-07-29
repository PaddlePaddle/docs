.. _cn_api_fluid_layers_sequence_concat:

sequence_concat
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_concat(input, name=None)

sequence_concat操作通过序列信息连接LoD张量（Tensor）。例如：X1的LoD = [0,3,7]，X2的LoD = [0,7,9]，结果的LoD为[0，（3 + 7），（7 + 9）]，即[0,10,16]。

参数:
        - **input** (list) – 要连接变量的列表
        - **name** (str|None) – 此层的名称(可选)。如果没有设置，该层将被自动命名。

返回:     连接好的输出变量。

返回类型:   变量（Variable）


**代码示例**

..  code-block:: python

        import paddle.fluid as fluid
        x = fluid.layers.data(name='x', shape=[10], dtype='float32')
        y = fluid.layers.data(name='y', shape=[10], dtype='float32')
        out = fluid.layers.sequence_concat(input=[x, y])










