.. _cn_api_fluid_layers_create_array:

create_array
-------------------------------

.. py:function:: paddle.fluid.layers.create_array(dtype)


创建LoDTensorArray数组。它主要用于实现RNN与array_write, array_read和While。

参数:
    - **dtype** (int |float) — lod_tensor_array中存储元素的数据类型。

返回: lod_tensor_array， 元素数据类型为dtype。

返回类型: Variable。


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  data = fluid.layers.create_array(dtype='float32')











