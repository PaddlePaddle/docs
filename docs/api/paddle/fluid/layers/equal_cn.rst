.. _cn_api_fluid_layers_equal:

equal
-------------------------------

.. py:function:: paddle.fluid.layers.equal(x, y, cond=None, name=None)


该OP返回 :math:`x==y` 逐元素比较x和y是否相等，x和y的维度应该相同。

参数
::::::::::::

    - **x** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64，int32， int64。
    - **y** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64， int32， int64。
    - **cond** (Variable，可选) – 如果为None，则创建一个Tensor来作为进行比较的输出结果，该Tensor的shape和数据类型和输入x一致；如果不为None，则将Tensor作为该OP的输出，数据类型和数据shape需要和输入x一致。默认值为None。 
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出结果的Tensor，输出Tensor的shape和输入一致，Tensor数据类型为bool。

返回类型
::::::::::::
变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.equal