.. _cn_api_fluid_layers_greater_equal:

greater_equal
-------------------------------

.. py:function:: paddle.fluid.layers.greater_equal(x, y, cond=None, name=None)




该 OP 逐元素地返回 :math:`x >= y` 的逻辑值，使用重载算子 `>=` 可以有相同的计算函数效果。


参数
::::::::::::

    - **x** (Variable) – 进行比较的第一个输入，是一个多维的 Tensor，数据类型可以是 float32，float64，int32，int64。
    - **y** (Variable) – 进行比较的第二个输入，是一个多维的 Tensor，数据类型可以是 float32，float64，int32，int64。
    - **cond** (Variable，可选) – 如果为 None，则创建一个 Tensor 来作为进行比较的输出结果，该 Tensor 的 shape，数据类型和输入 x 一致；如果不为 None，则将 Tensor 作为该 OP 的输出，数据 shape 和数据类型需要和输入 x 一致。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出结果的 Tensor，数据的 shape 和输入 x 一致。

返回类型
::::::::::::
Variable，数据类型为 bool 类型。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.greater_equal
