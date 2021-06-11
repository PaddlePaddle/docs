.. _cn_api_paddle_expm1:

expm1
-------------------------------

.. py:function:: paddle.expm1(x, name=None)




对输入，逐元素进行以自然数e为底指数运算并减1。

.. math::
    out = e^x - 1

参数
:::::::::

- **x** (Tensor) - 该OP的输入为多维Tensor。数据类型为：float16、float32、float64。
- **name** (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出为Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
:::::::::

.. code-block:: python

  import paddle

  x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
  out = paddle.expm1(x)
  print(out)
  # [-0.32967997, -0.18126924,  0.10517092,  0.34985882]
