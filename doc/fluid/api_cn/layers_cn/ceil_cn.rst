.. _cn_api_fluid_layers_ceil:

ceil
-------------------------------

.. py:function:: paddle.fluid.layers.ceil(x, name=None)

:alias_main: paddle.ceil
:alias: paddle.ceil,paddle.tensor.ceil,paddle.tensor.math.ceil
:old_api: paddle.fluid.layers.ceil



向上取整运算函数。

.. math::
    out = \left \lceil x \right \rceil



参数:
    - **x** (Variable) - 该OP的输入为多维Tensor。数据类型为float32或float64。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回： 输出为Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型： Variable

**代码示例**：

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  input_ceil = np.array([[-1.5,6],[1,15.6]])
  with fluid.dygraph.guard():
      x = fluid.dygraph.to_variable(input_ceil)
      y = fluid.layers.ceil(x)
      print(y.numpy())
      # [[-1.  6.]
      # [ 1. 16.]]
