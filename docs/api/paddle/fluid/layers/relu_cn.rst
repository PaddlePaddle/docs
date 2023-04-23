.. _cn_api_fluid_layers_relu:

relu
-------------------------------

.. py:function:: paddle.fluid.layers.relu(x, name=None)




ReLU（Rectified Linear Unit）激活函数

.. math:: Out=max(0,x)


参数
::::::::::::

  - **x** (Variable) - 输入的多维 ``Tensor``，数据类型为：float32、float64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 与 ``x`` 维度相同、数据类型相同的 ``Tensor`` 。

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.relu