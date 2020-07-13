.. _cn_api_fluid_layers_hash:

hash
-------------------------------

.. py:function::  paddle.fluid.layers.hash(input, hash_size, num_hash=1, name=None)

:alias_main: paddle.nn.functional.hash
:alias: paddle.nn.functional.hash,paddle.nn.functional.lod.hash
:old_api: paddle.fluid.layers.hash



该OP将输入 hash 成为一个整数，该数的值小于给定的 ``hash_size`` 。**仅支持输入为LoDTensor**。

该OP使用的哈希算法是：xxHash - `Extremely fast hash algorithm <https://github.com/Cyan4973/xxHash/tree/v0.6.5>`_


参数：
  - **input** (Variable) - 输入是一个 **二维** ``LoDTensor`` 。**输入维数必须为2**。数据类型为：int32、int64。**仅支持LoDTensor**。
  - **hash_size** (int) - 哈希算法的空间大小。输出值将保持在 :math:`[0, hash\_size)` 范围内。
  - **num_hash** (int) - 哈希次数。默认值为1。
  - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：``LoDTensor``

返回类型：Variable

**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  place = fluid.core.CPUPlace()

  # 构建网络
  x = fluid.data(name="x", shape=[2, 2], dtype="int32", lod_level=1)
  res = fluid.layers.hash(name="res", input=x, hash_size=1000, num_hash=4)

  # 创建CPU执行器
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())

  in1 = np.array([[1,2],[3,4]]).astype("int32")
  print(in1)
  x_i = fluid.create_lod_tensor(in1, [[0, 2]], place)
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
