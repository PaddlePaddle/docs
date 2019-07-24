.. _cn_api_fluid_layers_unsqueeze:

unsqueeze
-------------------------------

.. py:function:: paddle.fluid.layers.unsqueeze(input, axes, name=None)

向张量shape中插入一个维度。该接口接受axes列表，来指定要插入的维度位置。相应维度变化可以在输出变量中axes指定的索引位置上体现。

比如：
    给定一个张量，例如维度为[3,4,5]的张量，使用 axes列表为[0,4]来unsqueeze它，则输出维度为[1,3,4,5,1]

参数：
    - **input** (Variable)- 未压缩的输入变量
    - **axes** (list)- 一列整数，代表要插入的维数
    - **name** (str|None) - 该层名称

返回：输出未压缩变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[5, 10])
    y = fluid.layers.unsequeeze(input=x, axes=[1])










