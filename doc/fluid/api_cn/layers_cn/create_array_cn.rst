.. _cn_api_fluid_layers_create_array:

create_array
-------------------------------

.. py:function:: paddle.fluid.layers.create_array(dtype)


此OP创建一个LoDTensorArray数组, 它主要用于实现RNN与array_write, array_read和While OP。

参数:
    - **dtype** (int |float) — 指定Tensor中元素的数据类型，支持的数据类型有int32，int64，float32和float64。

返回: 返回创建的LoDTensor Array，Tensor中的元素数据类型为指定的dtype。

返回类型: Variable。


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  data = fluid.layers.create_array(dtype='float32') # 创建一个数据类型为float32的LoDTensor Array。











