.. _cn_api_fluid_layers_cos:

cos
-------------------------------

.. py:function:: paddle.fluid.layers.cos(x, name=None)

:alias_main: paddle.cos
:alias: paddle.cos,paddle.tensor.cos,paddle.tensor.math.cos
:old_api: paddle.fluid.layers.cos



余弦函数。

.. math::

    out = cos(x)



参数:
    - **x** (Variable) - 该OP的输入为多维Tensor，数据类型为float32，float64。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。


返回：输出为Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型：Variable

**代码示例**：

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  input_cos = np.array([[-1,np.pi],[1,15.6]])
  with fluid.dygraph.guard():
      x = fluid.dygraph.to_variable(input_cos)
      y = fluid.layers.cos(x)
      print(y.numpy())
      # [[ 0.54030231 -1.        ]
      # [ 0.54030231 -0.99417763]]
