.. _cn_api_paddle_tensor_put_along_axis:

put_along_axis
-------------------------------

.. py:function:: paddle.put_along_axis(arr, indices, value, axis, reduce)

参数
:::::::::

- **arr**  (Tensor) - 输入的Tensor 作为目标矩阵，数据类型为：float32、float64。
- **indices**  (Tensor) - 索引矩阵，包含沿轴提取1d切片的下标, 必须和arr矩阵有相同的维度, 需要能够broadcast与arr矩阵对齐, 数据类型为: int、int64。
- **value** 需要插入的值, 形状和维度需要能够被broadcast与indices矩阵匹配。
- **axis**  (int) - 指定沿着哪个维度获取对应的值, 数据类型为: int.
- **reduce** (str|可选) - 归约操作类型，默认为'assign', 可选为 ‘add' 或 ‘multiple’。

返回
:::::::::

- **out** (Tensor) - 输出Tensor，indeces矩阵选定的下标会被插入value, 与 ``arr`` 数据类型相同。

代码示例
:::::::::

.. code-block:: python

      import paddle
      import numpy as np

      x_np = np.array([[10, 30, 20], [60, 40, 50]])
      index_np = np.array([[0]])
      x = paddle.to_tensor(x_np)
      index = paddle.to_tensor(index_np)
      value = 99
      axis = 0
      result = paddle.put_along_axis(x, index, value, axis)
      print(result)
      # [[99, 99, 99],
      # [60, 40, 50]]