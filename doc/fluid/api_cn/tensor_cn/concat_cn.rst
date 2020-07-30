.. _cn_api_tensor_concat:

concat
-------------------------------

.. py:function:: paddle.tensor.concat(x, axis=0, name=None)

:alias_main: paddle.concat
:alias: paddle.tensor.concat, paddle.tensor.manipulation.concat


该OP对输入沿 ``axis`` 轴进行联结，返回一个新的Tensor。

参数：
    - **x** (list) - 待联结的Tensor list ，支持的数据类型为：float16, float32、float64、int32、int64， ``x`` 中的所有数据类型应该一致。
    - **axis** (int|Tensor，可选) - 指定对输入 ``x`` 进行运算的轴，数据类型为整数或者形状为[1]的 Tensor，数据类型为int32或者int64。 ``axis`` 的有效范围是[-R, R)，R是输入 ``x`` 中Tensor的维度， ``axis`` 为负值时与 :math:`axis + R` 等价。默认值为0。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：联结后的Tensor ，数据类型和 ``x`` 中的Tensor相同。

抛出异常：
    - ``TypeError``: - 当输入 ``x`` 的数据类型不是 float16， float32， float64， int32， int64时。
    - ``TypeError``: - 当 ``axis`` 的数据类型不是int或者Tensor时。 当 ``axis`` 是Tensor的时候其数据类型不是int32或者int64时。
    - ``TypeError``: - 当输入 ``x`` 中所有Tensor存在数据类型不一致时。

**代码示例**：

.. code-block:: python
  
  import paddle
  import numpy as np
  
  paddle.enable_imperative()  # Now we are in imperative mode
  in1 = np.array([[1,2,3],
                  [4,5,6]])
  in2 = np.array([[11,12,13],
                  [14,15,16]])
  in3 = np.array([[21,22],
                  [23,24]])
  x1 = paddle.imperative.to_variable(in1)
  x2 = paddle.imperative.to_variable(in2)
  x3 = paddle.imperative.to_variable(in3)
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
