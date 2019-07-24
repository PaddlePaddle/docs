.. _cn_api_fluid_layers_array_length:

array_length
-------------------------------

.. py:function:: paddle.fluid.layers.array_length(array)

**得到输入LoDTensorArray的长度**

此功能用于查找输入数组LOD_TENSOR_ARRAY的长度。

相关API:
    - :ref:`cn_api_fluid_layers_array_read`
    - :ref:`cn_api_fluid_layers_array_write`
    - :ref:`cn_api_fluid_layers_While`

参数：
    - **array** (LOD_TENSOR_ARRAY)-输入数组，用来计算数组长度

返回：输入数组LoDTensorArray的长度

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    arr = fluid.layers.array_write(tmp, i=i)
    arr_len = fluid.layers.array_length(arr)









