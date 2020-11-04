.. _cn_api_fluid_layers_log2:

log2
-------------------------------

.. py:function:: paddle.fluid.layers.log2(x, name=None)





Log2激活函数（计算底为2的对数）

.. math::
                  \\Out=log_2x\\


参数:
  - **x** (Tensor) – 该OP的输入为Tensor。数据类型为float32，float64。 
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：Log2算子底为2对数输出

返回类型: Tensor - 该OP的输出为Tensor，数据类型为输入一致。


**代码示例**

..  code-block:: python

  import paddle
  import paddle.fluid as fluid
  
  # example 1: x is a float
  x_i = paddle.to_tensor([[1.0], [2.0]])
  res = paddle.log2(x_i) # [[0.], [1.0]]

  # example 2: x is float32
  x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
  paddle.to_tensor(x_i)
  res = paddle.log2(x_i)
  print(res.numpy()) # [1.]
  
  # example 3: x is float64
  x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
  paddle.to_tensor(x_i)
  res = paddle.log2(x_i)
  print(res.numpy()) # [1.]
