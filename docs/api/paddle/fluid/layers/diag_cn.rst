.. _cn_api_fluid_layers_diag:

diag
-------------------------------

.. py:function:: paddle.fluid.layers.diag(diagonal)




该 OP 创建一个方阵，使用输入 diagonal 来指定方阵的对角线元素的值。

参数
::::::::::::

    - **diagonal** (Variable|numpy.ndarray) — 数据 shape 为 :math:`[N]` 一维 Tensor，会把该 Tensor 的元素赋在方阵的对角线上。数据类型可以是 float32，float64，int32，int64。

返回
::::::::::::
存储着方阵的 Tensor，对角线值是输入 Tensor diagonal 的值，数据 shape 为 :math:`[N, N]` 二维 Tensor。

返回类型
::::::::::::
Variable，数据类型和输入数据类型一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.diag
