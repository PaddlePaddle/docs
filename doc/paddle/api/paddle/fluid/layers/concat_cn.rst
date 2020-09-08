.. _cn_api_fluid_layers_concat:

concat
-------------------------------

.. py:function:: paddle.fluid.layers.concat(input, axis=0, name=None)


该OP对输入沿 ``axis`` 轴进行联结，返回一个新的Tensor。

参数：
    - **input** (list|tuple|Tensor) - 待联结的Tensor list，Tensor tuple或者Tensor，支持的数据类型为：bool、float16、 float32、float64、int32、int64。 ``input`` 中所有Tensor的数据类型必须一致。
    - **axis** (int|Tensor，可选) - 指定对输入Tensor进行运算的轴，可以是整数或者形状为[1]的Tensor，数据类型为int32或者int64。 ``axis`` 的有效范围是[-R, R)，R是输入 ``input`` 中Tensor 的维度， ``axis`` 为负值时与 :math:`axis + R` 等价。默认值为0。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：联结后的 ``Tensor`` ，数据类型和 ``input`` 中的Tensor相同。

**代码示例**：

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  in1 = np.array([[1, 2, 3],
                  [4, 5, 6]])
  in2 = np.array([[11, 12, 13],
                  [14, 15, 16]])
  in3 = np.array([[21, 22],
                  [23, 24]])
  with fluid.dygraph.guard():
      x1 = fluid.dygraph.to_variable(in1)
      x2 = fluid.dygraph.to_variable(in2)
      x3 = fluid.dygraph.to_variable(in3)
      out1 = fluid.layers.concat(input=[x1, x2, x3], axis=-1)
      out2 = fluid.layers.concat(input=[x1, x2], axis=0)
      print(out1.numpy())
      # [[ 1  2  3 11 12 13 21 22]
      #  [ 4  5  6 14 15 16 23 24]]
      print(out2.numpy())
      # [[ 1  2  3]
      #  [ 4  5  6]
      #  [11 12 13]
      #  [14 15 16]]
