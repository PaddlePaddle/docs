.. _cn_api_fluid_layers_equal:

equal
-------------------------------

.. py:function:: paddle.fluid.layers.equal(x, y, cond=None, name=None)


该 OP 返回 :math:`x==y` 逐元素比较 x 和 y 是否相等，x 和 y 的维度应该相同。

参数
::::::::::::

    - **x** (Variable) - 输入 Tensor，支持的数据类型包括 float32， float64，int32， int64。
    - **y** (Variable) - 输入 Tensor，支持的数据类型包括 float32， float64， int32， int64。
    - **cond** (Variable，可选) – 如果为 None，则创建一个 Tensor 来作为进行比较的输出结果，该 Tensor 的 shape 和数据类型和输入 x 一致；如果不为 None，则将 Tensor 作为该 OP 的输出，数据类型和数据 shape 需要和输入 x 一致。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出结果的 Tensor，输出 Tensor 的 shape 和输入一致，Tensor 数据类型为 bool。

返回类型
::::::::::::
变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.equal
