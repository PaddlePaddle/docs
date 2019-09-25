.. _cn_api_fluid_layers_cast:

cast
-------------------------------

.. py:function:: paddle.fluid.layers.cast(x,dtype)

该OP将 ``x`` 的数据类型转换为 ``dtype`` 并输出。支持输出和输入的数据类型相同。

参数：
    - **x** (Variable) - 输入的多维Tensor或LoDTensor，支持的数据类型为：bool、float16、float32、float64、uint8、int32、int64。
    - **dtype** (str|np.dtype|core.VarDesc.VarType) - 输出Tensor的数据类型。支持的数据类型为：bool、float16、float32、float64、int8、int32、int64、uint8。

返回：Tensor或LoDTensor，维度与 ``x`` 相同，数据类型为 ``dtype``

返回类型：Variable

**代码示例**：

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  place = fluid.core.CPUPlace()

  # 构建网络
  x_lod = fluid.layers.data(name="x", shape=[1], lod_level=1)
  cast_res1 = fluid.layers.cast(x=x_lod, dtype="uint8")
  cast_res2 = fluid.layers.cast(x=x_lod, dtype=np.int32)

  # 创建CPU执行器
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())

  x_i_lod = fluid.core.LoDTensor()
  x_i_lod.set(np.array([[1.3,-2.4],[0,4]]).astype("float32"), place)
  x_i_lod.set_recursive_sequence_lengths([[0,2]])
  res1 = exe.run(fluid.default_main_program(), feed={'x':x_i_lod}, fetch_list=[cast_res1], return_numpy=False)
  res2 = exe.run(fluid.default_main_program(), feed={'x':x_i_lod}, fetch_list=[cast_res2], return_numpy=False)
  print(np.array(res1[0]), np.array(res1[0]).dtype)
  # [[  1 254]
  #  [  0   4]] uint8
  print(np.array(res2[0]), np.array(res2[0]).dtype)
  # [[ 1 -2]
  #  [ 0  4]] int32
