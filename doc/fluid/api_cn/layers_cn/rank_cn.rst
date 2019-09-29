.. _cn_api_fluid_layers_rank:

rank
-------------------------------

.. py:function::  paddle.fluid.layers.rank(input)

该OP用于计算输入Tensor的维度（秩）。

参数：
    - **input** (Variable) — 输入input是shape为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor，数据类型可以任意类型。

返回：输出Tensor的秩，是一个0-D Tensor。

返回类型：Variable，数据类型为int32。

**代码示例**

.. code-block:: python

       import paddle.fluid as fluid
       input = fluid.layers.data(
            name="input", shape=[3, 100, 100], dtype="float32")
       rank = fluid.layers.rank(input) # rank=(3,)


