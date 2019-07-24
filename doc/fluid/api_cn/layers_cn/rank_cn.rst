.. _cn_api_fluid_layers_rank:

rank
-------------------------------

.. py:function::  paddle.fluid.layers.rank(input)

排序层

返回张量的维数，一个数据类型为int32的0-D Tensor。

参数:
    - **input** (Variable)：输入变量

返回：输入变量的秩

返回类型： 变量（Variable）

**代码示例**

.. code-block:: python

       import paddle.fluid as fluid
       input = layers.data(
            name="input", shape=[3, 100, 100], dtype="float32")
       rank = layers.rank(input) # 4


