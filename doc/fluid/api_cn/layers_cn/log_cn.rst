.. _cn_api_fluid_layers_log:

log
-------------------------------

.. py:function:: paddle.fluid.layers.log(x, name=None)


Log激活函数（计算自然对数）

.. math::
                  \\Out=ln(x)\\


参数:
  - **x** (Variable) – 该OP的输入为LodTensor/Tensor。数据类型为float32，float64。 
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：Log算子自然对数输出

返回类型: Variable - 该OP的输出为LodTensor/Tensor，数据类型为输入一致。


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  # Graph Organizing
  x = fluid.layers.data(name="x", shape=[1], dtype="float32")
  res = fluid.layers.log(x)
  
  # Create an executor using CPU as an example
  exe = fluid.Executor(fluid.CPUPlace())
  exe.run(fluid.default_startup_program())

  # Execute
  x_i = np.array([[1], [2]]).astype(np.float32)
  res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
  print(res_val) # [[0.], [0.6931472]]

