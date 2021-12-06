.. _cn_api_paddle_tensor_deg2rad:

deg2rad
-------------------------------

.. py:function:: paddle.deg2rad(x, name=None)

将元素从弧度的角度转换为度

.. math::

    deg2rad(x)=\pi * x / 180

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
  
  x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
  result1 = paddle.deg2rad(x1)
  print(result1)
  # Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
  #         [3.14159274, -3.14159274,  6.28318548, -6.28318548,  1.57079637,
  #           -1.57079637])

  x2 = paddle.to_tensor(180)
  result2 = paddle.deg2rad(x2)
  print(result2)
  # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
  #         [3.14159274])
