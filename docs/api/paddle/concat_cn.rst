.. _cn_api_tensor_concat:

concat
-------------------------------

.. py:function:: paddle.concat(x, axis=0, name=None)


该OP对输入沿 ``axis`` 轴进行联结，返回一个新的Tensor。

参数：
    - **x** (list|tuple) - 待联结的Tensor list或者Tensor tuple ，支持的数据类型为：bool, float16, float32、float64、int32、int64， ``x`` 中所有Tensor的数据类型应该一致。
    - **axis** (int|Tensor，可选) - 指定对输入 ``x`` 进行运算的轴，可以是整数或者形状为[1]的Tensor，数据类型为int32或者int64。 ``axis`` 的有效范围是[-R, R)，R是输入 ``x`` 中Tensor的维度， ``axis`` 为负值时与 :math:`axis + R` 等价。默认值为0。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：联结后的Tensor ，数据类型和 ``x`` 中的Tensor相同。


**代码示例**：

.. code-block:: python
  
  import paddle
  import numpy as np
  
  in1 = np.array([[1, 2, 3],
                  [4, 5, 6]])
  in2 = np.array([[11, 12, 13],
                  [14, 15, 16]])
  in3 = np.array([[21, 22],
                  [23, 24]])
  x1 = paddle.to_tensor(in1)
  x2 = paddle.to_tensor(in2)
  x3 = paddle.to_tensor(in3)
  zero = paddle.full(shape=[1], dtype='int32', fill_value=0)
  # When the axis is negative, the real axis is (axis + Rank(x))
  # As follow, axis is -1, Rank(x) is 2, the real axis is 1
  out1 = paddle.concat(x=[x1, x2, x3], axis=-1)
  out2 = paddle.concat(x=[x1, x2], axis=0)
  out3 = paddle.concat(x=[x1, x2], axis=zero)
  # out1
  # [[ 1  2  3 11 12 13 21 22]
  #  [ 4  5  6 14 15 16 23 24]]
  # out2 out3
  # [[ 1  2  3]
  #  [ 4  5  6]
  #  [11 12 13]
  #  [14 15 16]]
