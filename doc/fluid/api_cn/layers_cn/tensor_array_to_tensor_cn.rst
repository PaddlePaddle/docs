.. _cn_api_fluid_layers_tensor_array_to_tensor:

tensor_array_to_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.tensor_array_to_tensor(input, axis=1, name=None)

该OP在指定轴上连接LoDTensorArray中的元素。

参数：
  - **input** (Variable) - 输入的LoDTensorArray。支持的数据类型为：float32、float64、int32、int64。
  - **axis** (int，可选) - 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-R, R)，R是输入 ``input`` 中Tensor的Rank，``axis`` 为负时与 ``axis`` +R 等价。默认值为1。
  - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：LoDTensor

返回类型： Variable

**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  place = fluid.CPUPlace()

  x1 = fluid.layers.data(name="x", shape=[2,2], lod_level=0)
  tmp = fluid.layers.fill_constant(shape=[2,3], dtype="float32", value=1)
  x_arr = fluid.layers.create_array(dtype="float32")
  c0 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
  fluid.layers.array_write(x=tmp, i=c0, array=x_arr)
  c1 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
  fluid.layers.array_write(x=x1, i=c1, array=x_arr)
  output, output_index = fluid.layers.tensor_array_to_tensor(input=x_arr, axis=1)

  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())

  feedx = fluid.LoDTensor()
  feedx.set(np.array([[1.3,-2.4],[0,4]]).astype("float32"), place)
  res = exe.run(fluid.default_main_program(), feed={'x':feedx}, fetch_list=[output], return_numpy=False)

  print(np.array(res[0]))
  # [[ 1.   1.   1.   1.3 -2.4]
  #  [ 1.   1.   1.   0.   4. ]]
