.. _cn_api_paddle_tensor_rad2deg:

rad2deg
-------------------------------

.. py:function:: paddle.rad2deg(x, name=None)

将元素从弧度的角度转换为度

.. math::

    rad2deg(x)=180/ \pi * x

参数
:::::::::

- **x**  (Tensor) - 输入的Tensor，数据类型为：int32、int64、float32、float64。
- **name**  (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出Tensor，与 ``x`` 维度相同、数据类型相同（输入为int时，输出数据类型为float32）。

代码示例
:::::::::

.. code-block:: python

  import paddle
  import numpy as np
  
  x1 = paddle.to_tensor([3.142, -3.142, 6.283, -6.283, 1.570, -1.570])
  result1 = paddle.rad2deg(x1)
  print(result1)
  # Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
  #         [180.02334595, -180.02334595,  359.98937988, -359.98937988,
  #           9.95437622 , -89.95437622])

  x2 = paddle.to_tensor(np.pi/2)
  result2 = paddle.rad2deg(x2)
  print(result2)
  # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
  #         [90.])
           
  x3 = paddle.to_tensor(1)
  result3 = paddle.rad2deg(x3)
  print(result3)
  # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
  #         [57.29578018])
