.. _cn_api_fluid_layers_array_length:

array_length
-------------------------------

.. py:function:: paddle.fluid.layers.array_length(array)

该OP用于获取输入数组 :ref:`cn_api_fluid_LoDTensorArray` 的长度。可以与 :ref:`cn_api_fluid_layers_array_read` 、 :ref:`cn_api_fluid_layers_array_write` 、 :ref:`cn_api_fluid_layers_While` OP结合使用，实现LoDTensorArray的遍历与读写。

参数：
    - **array** (LoDTensorArray) - 输入的数组LoDTensorArray

返回：shape为[1]的1-D Tensor, 表示数组LoDTensorArray的长度，数据类型为int64

返回类型：Variable

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    arr = fluid.layers.array_write(tmp, i=i)
    arr_len = fluid.layers.array_length(arr)









