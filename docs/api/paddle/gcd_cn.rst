.. _cn_api_paddle_tensor_gcd:

gcd
-------------------------------

.. py:function:: paddle.gcd(x, y, name=None)

计算两个输入的按元素绝对值的最大公约数，输入必须是整型。

.. note::

    gcd(0,0)=0, gcd(0, y)=|y|

参数
:::::::::

- **x, y**  (Tensor) - 输入的Tensor，数据类型为：int8，int16，int32，int64，uint8。
    如果x和y的shape不一致，会对两个shape进行广播操作，得到一致的shape（并作为输出结果的shape），
    请参见 :ref:`cn_user_guide_broadcasting` 。
- **name**  (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出Tensor，与输入数据类型相同。

代码示例
:::::::::

.. code-block:: python

  import paddle
  import numpy as np
  
  x1 = paddle.to_tensor(12)
  x2 = paddle.to_tensor(20)
  paddle.gcd(x1, x2)
  # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
  #        [4])

  x3 = paddle.to_tensor(np.arange(6))
  paddle.gcd(x3, x2)
  # Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
  #        [20, 1 , 2 , 1 , 4 , 5])

  x4 = paddle.to_tensor(0)
  paddle.gcd(x4, x2)
  # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
  #        [20])

  paddle.gcd(x4, x4)
  # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
  #        [0])
  
  x5 = paddle.to_tensor(-20)
  paddle.gcd(x1, x5)
  # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
  #        [4])
