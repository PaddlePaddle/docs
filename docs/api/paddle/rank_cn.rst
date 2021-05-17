.. _cn_api_fluid_layers_rank:

rank
-------------------------------

.. py:function::  paddle.rank(input)




该OP用于计算输入Tensor的维度（秩）。

参数：
    - **input** (Tensor) — 输入input是shape为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor，数据类型可以任意类型。

返回：输出Tensor的秩，是一个0-D Tensor。


**代码示例**

.. code-block:: python

       import paddle
       input = paddle.rand((3, 100, 100))
       rank = paddle.rank(input)