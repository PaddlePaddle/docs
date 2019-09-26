.. _cn_api_fluid_layers_array_write:

array_write
-------------------------------

.. py:function:: paddle.fluid.layers.array_write(x, i, array=None)

该OP将输入的变量 ``x`` 写入到数组 :ref:`LoDTensorArray` ``array`` 的第i个位置，并返回修改后的LoDTensorArray，如果 ``array`` 为None，则创建一个新的LoDTensorArray。

参数:
    - **x** (Variable) – 待写入的数据，多维Tensor或LoDTensor
    - **i** (Variable) – shape为[1]的1-D Tensor，表示写入到输出数组LoDTensorArray的位置，数据类型为int64
    - **array** (Variable，可选) – 指定写入 ``x`` 的数组LoDTensorArray。默认值为None, 此时将创建新的LoDTensorArray并作为结果返回

返回: 写入输入 ``x`` 之后的LoDTensorArray

返回类型: Variable

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  tmp = fluid.layers.zeros(shape=[10], dtype='int32')
  i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
  arr = fluid.layers.array_write(tmp, i=i)











