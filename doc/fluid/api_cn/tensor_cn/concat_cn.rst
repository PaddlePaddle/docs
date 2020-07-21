.. _cn_api_tensor_concat:

concat
-------------------------------

.. py:function:: paddle.tensor.concat(x,axis=0,name=None)


该OP对输入沿 ``axis`` 轴进行联结，返回一个新的张量。

参数：
    - **x** (list) - 输入是待联结的多维 ``Tensor`` 组成的 ``list`` ，支持的数据类型为：float32、float64、int32、int64。
    - **axis** (int|Variable，可选) - 整数或者形状为[1]的 ``Tensor``，数据类型为 ``int32``。指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-R, R)，R是输入 ``x`` 中 ``Tensor`` 的维度， ``axis`` 为负值时与 :math:`axis + R` 等价。默认值为0。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：联结后的 ``Tensor`` ，数据类型和 ``x`` 相同。

返回类型：Variable

抛出异常：
    - ``TypeError``: - 如果输入的数据类型不是 float32， float64， int32， int64其中之一。

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
  out1 = paddle.concat(x=[x1,x2,x3], axis=-1)
  out2 = paddle.concat(x=[x1,x2], axis=0)
  out2 = paddle.concat(x=[x1,x2], axis=zero)
  # out1
  # [[ 1  2  3 11 12 13 21 22]
  #  [ 4  5  6 14 15 16 23 24]]
  # out2 out3
  # [[ 1  2  3]
  #  [ 4  5  6]
  #  [11 12 13]
  #  [14 15 16]]
