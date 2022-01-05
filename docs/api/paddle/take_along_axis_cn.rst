.. _cn_api_paddle_tensor_take_along_axis:

take_along_axis
-------------------------------

.. py:function:: paddle.take_along_axis(arr, indices, axis)
基于输入索引矩阵, 沿着指定axis从arr矩阵里选取1d切片。索引矩阵必须和arr矩阵有相同的维度, 需要能够broadcast与arr矩阵对齐。

参数
:::::::::

- **arr**  (Tensor) - 输入的Tensor 作为源矩阵，数据类型为：float32、float64。
- **indices**  (Tensor) - 索引矩阵，包含沿轴提取1d切片的下标, 必须和arr矩阵有相同的维度, 需要能够broadcast与arr矩阵对齐, 数据类型为: int、int64。
- **axis**  (int) - 指定沿着哪个维度获取对应的值, 数据类型为: int。

返回
:::::::::

- **out** (Tensor) - 输出Tensor，包含indeces矩阵选定的元素, 与 ``arr`` 数据类型相同。

代码示例
:::::::::

.. code-block:: python

      import paddle
      import numpy as np

      x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7,8,9]])
      index = paddle.to_tensor([[0]])
      axis = 0
      result = paddle.take_along_axis(x, index, axis)
      print(result)
      # [[1, 2, 3]]