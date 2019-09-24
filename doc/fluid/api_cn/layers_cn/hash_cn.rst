.. _cn_api_fluid_layers_hash:

hash
-------------------------------

.. py:function::  paddle.fluid.layers.hash(input, hash_size, num_hash=1, name=None)

该OP将输入 hash 到一个整数，该数的值小于给定的 ``hash_size`` 。**仅支持输入为LodTensor**。

该OP使用的哈希算法是：xxHash - `Extremely fast hash algorithm <https://github.com/Cyan4973/xxHash/tree/v0.6.5>`_


参数：
  - **input** (Variable) - 输入是一个 **二维** ``LodTensor`` 。**输入维数必须为2**。数据类型为：int32、int64。**仅支持LodTensor**。
  - **hash_size** (int) - 哈希算法的空间大小。输出值将保持在 :math:`[0, hash\_size)` 范围内。
  - **num_hash** (int) - 哈希次数。默认值为1。
  - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回：哈希的结果变量，与输入变量类型相同。

返回类型： Variable

**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  place = fluid.core.CPUPlace()

  # Graph Organizing
  x = fluid.layers.data(name="x", shape=[1], dtype="int32", lod_level=1)
  res = fluid.layers.hash(name="res",input=x, hash_size=1000, num_hash=4)

  # Create an executor using CPU as an example
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())

  in1 = np.array([[1,2],[3,4]]).astype("int32")
  print(in1)
  x_i = fluid.core.LoDTensor()
  x_i.set(in1,place)
  x_i.set_recursive_sequence_lengths([[0,2]])
  res = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res], return_numpy=False)
  print(np.array(res[0]))
  # [[[722]
  #   [407]
  #   [337]
  #   [395]]
  #  [[603]
  #   [590]
  #   [386]
  #   [901]]]
