.. _cn_api_fluid_layers_exp:

exp
-------------------------------

.. py:function:: paddle.exp(x, name=None)




对输入，逐元素进行以自然数e为底指数运算。

.. math::
    out = e^x

参数:
    - **x** (Tensor) - 该OP的输入为多维Tensor。数据类型为float32，float64。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回：输出为Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型： Tensor

**代码示例**：

.. code-block:: python

  import paddle

  x = paddle.to_tensor([[-1.5,6],[1,15.6]])
  y = paddle.exp(x)
  print(y)
  # [[2.23130160e-01 4.03428793e+02]
  # [2.71828183e+00 5.95653801e+06]]

