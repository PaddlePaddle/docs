.. _cn_api_fluid_layers_array_write:

array_write
-------------------------------

.. py:function:: paddle.fluid.layers.array_write(x, i, array=None)


该函数将给定的输入变量（即 ``x`` ）写入一个作为输出的 ``LOD_TENSOR_ARRAY`` 变量的某一指定位置中，
这一位置由数组下标(即 ``i`` )指明。 如果 ``LOD_TENSOR_ARRAY`` (即 ``array`` )未指定（即为None值）， 一个新的 ``LOD_TENSOR_ARRAY`` 将会被创建并作为结果返回。

参数:
    - **x** (Variable|list) – 待从中读取数据的输入张量(tensor)
    - **i** (Variable|list) – 输出结果 ``LOD_TENSOR_ARRAY`` 的下标, 该下标指向输入张量 ``x`` 写入输出数组的位置
    - **array** (Variable|list) – 会被输入张量 ``x`` 写入的输出结果 ``LOD_TENSOR_ARRAY`` 。如果该项值为None， 一个新的 ``LOD_TENSOR_ARRAY`` 将会被创建并作为结果返回

返回: 输入张量 ``x`` 所写入的输出结果 ``LOD_TENSOR_ARRAY``

返回类型: 变量（Variable）

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  tmp = fluid.layers.zeros(shape=[10], dtype='int32')
  i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
  arr = fluid.layers.array_write(tmp, i=i)











