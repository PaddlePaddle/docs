.. _cn_api_fluid_layers_relu6:

relu6
-------------------------------

.. py:function:: paddle.fluid.layers.relu6(x, threshold=6.0, name=None)

relu6激活函数

.. math:: out=min(max(0, x), threshold)


参数:
    - **x** (Variable) - 输入的多维 ``Tensor`` ，数据类型为：float32、float64。
    - **threshold** (float) - relu6的阈值。默认值为6.0
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回: 与 ``x`` 维度相同、数据类型相同的 ``Tensor``。

返回类型: Variable


**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  in1 = np.array([[-1,0],[2.5,7.8]])
  with fluid.dygraph.guard():
      x1 = fluid.dygraph.to_variable(in1)
      out1 = fluid.layers.relu6(x=x1, threshold=6.0)
      print(out1.numpy())
      # [[0.  0. ]
      #  [2.5 6. ]]
