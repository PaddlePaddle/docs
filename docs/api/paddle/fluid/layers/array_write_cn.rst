.. _cn_api_fluid_layers_array_write:

array_write
-------------------------------

.. py:function:: paddle.fluid.layers.array_write(x, i, array=None)




该OP将输入的变量 ``x`` 写入到数组 :ref:`cn_api_fluid_LoDTensorArray` ``array`` 的第i个位置，并返回修改后的LoDTensorArray，如果 ``array`` 为None，则创建一个新的LoDTensorArray。常与 :ref:`cn_api_fluid_layers_array_read` OP联合使用对LoDTensorArray进行读写。

参数
::::::::::::

    - **x** (Variable) – 待写入的数据，多维Tensor或LoDTensor，数据类型支持float32，float64，int32，int64
    - **i** (Variable) – shape为[1]的1-D Tensor，表示写入到输出数组LoDTensorArray的位置，数据类型为int64
    - **array** (Variable，可选) – 指定写入 ``x`` 的数组LoDTensorArray。默认值为None，此时将创建新的LoDTensorArray并作为结果返回

返回
::::::::::::
 写入输入 ``x`` 之后的LoDTensorArray

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

.. code-block:: python

  import paddle.fluid as fluid
  tmp = fluid.layers.fill_constant(shape=[3, 2], dtype='int64', value=5)
  i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
  #将tmp写入数组arr下标为10的位置，并返回arr
  arr = fluid.layers.array_write(tmp, i=i)

  #此时arr是长度为11的LoDTensorArray，可以通过array_read来查看下标为10的LoDTensor，并将之打印出来
  item = fluid.layers.array_read(arr, i=i)
  input = fluid.layers.Print(item, message="The content of i-th LoDTensor:")
  main_program = fluid.default_main_program()
  exe = fluid.Executor(fluid.CPUPlace())
  exe.run(main_program)

**输出结果**

COPY-FROM: paddle.fluid.layers.array_write