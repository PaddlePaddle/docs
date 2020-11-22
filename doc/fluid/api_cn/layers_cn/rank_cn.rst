.. _cn_api_fluid_layers_rank:

rank
-------------------------------

.. py:function::  paddle.fluid.layers.rank(input)

:old_api: paddle.fluid.layers.rank



该OP用于计算输入Tensor的维度。

参数：
    - **input** (Tensor) — 输入input是shape为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor，数据类型可以任意类型。

返回：输出Tensor的维度，是一个0-D Tensor。

返回类型：Tensor，数据类型为int32。

**代码示例**

.. code-block:: python

            import paddle

            input = paddle.rand((3, 100, 100))
            rank = paddle.rank(input)
            print(rank)
            # Tensor(shape=[], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
            #        3)
