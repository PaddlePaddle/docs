.. _cn_api_fluid_layers_create_array:

create_array
-------------------------------

.. py:function:: paddle.fluid.layers.create_array(dtype)





此 OP 创建一个 LoDTensorArray，它可以用作 :ref:`cn_api_fluid_layers_array\_write` , :ref:`cn_api_fluid_layers_array\_read` OP 的输入，以及和 :ref:`cn_api_fluid_layers_While` OP
一起创建 RNN 网络。

参数
::::::::::::

    - **dtype** (str) — 指定 Tensor 中元素的数据类型，支持的数据类型值：float32，float64，int32，int64。

返回
::::::::::::
 返回创建的空 LoDTensorArray，Tensor 中的元素数据类型为指定的 dtype。

返回类型
::::::::::::
 Variable。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.create_array
