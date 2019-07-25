.. _cn_api_fluid_layers_thresholded_relu:

thresholded_relu
-------------------------------

.. py:function:: paddle.fluid.layers.thresholded_relu(x,threshold=None)

ThresholdedRelu激活函数

.. math::

  out = \left\{\begin{matrix}
      x, &if x > threshold\\
      0, &otherwise
      \end{matrix}\right.

参数：
- **x** -ThresholdedRelu激活函数的输入
- **threshold** (FLOAT)-激活函数threshold的位置。[默认1.0]。

返回：ThresholdedRelu激活函数的输出

**代码示例**：

.. code-block:: python

  import paddle.fluid as fluid
  data = fluid.layers.data(name="input", shape=[1])
  result = fluid.layers.thresholded_relu(data, threshold=0.4)









