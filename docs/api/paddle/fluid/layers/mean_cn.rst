.. _cn_api_fluid_layers_mean:

mean
-------------------------------

.. py:function:: paddle.fluid.layers.mean(x, name=None)




计算 ``x`` 所有元素的平均值。

参数
::::::::::::

        - **x** (Variable) : Tensor。均值运算的输入。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

        - Variable：包含输出均值的 Tensor / LoDTensor。

返回类型
::::::::::::

        - Variable（变量）。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.mean
