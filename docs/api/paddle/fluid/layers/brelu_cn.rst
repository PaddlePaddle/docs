.. _cn_api_fluid_layers_brelu:

brelu
-------------------------------

.. py:function:: paddle.fluid.layers.brelu(x, t_min=0.0, t_max=24.0, name=None)





BReLU 激活函数

.. math::   out=min(max(x,t\_min),t\_max)

参数
::::::::::::

  - **x** (Variable) - 该OP的输入为多维Tensor。数据类型为float32，float64。
  - **t_min** (float，可选) - BRelu的最小值，默认值为0.0。
  - **t_max** (float，可选) - BRelu的最大值，默认值为24.0。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 输出为Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型
::::::::::::
 Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.brelu