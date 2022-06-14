.. _cn_api_paddle_atan2:

atan2
-------------------------------

.. py:function:: paddle.atan2(x, y, name=None)




对x/y进行逐元素的arctangent运算，通过符号确定象限

.. math::
    atan2(x,y)=\left\{\begin{matrix}
    & tan^{-1}(\frac{x}{y}) & y > 0 \\
    & tan^{-1}(\frac{x}{y}) + \pi & x>=0, y < 0 \\
    & tan^{-1}(\frac{x}{y}) - \pi & x<0, y < 0 \\
    & +\frac{\pi}{2} & x>0, y = 0 \\
    & -\frac{\pi}{2} & x<0, y = 0 \\
    &\text{undefined} & x=0, y = 0
    \end{matrix}\right.

参数
:::::::::

- **x**  (Tensor) - 输入的Tensor，数据类型为：int32、int64、float16、float32、float64。
- **y**  (Tensor) - 输入的Tensor，数据类型为：int32、int64、float16、float32、float64。
- **name**  (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见  :ref:`api_guide_Name`。

返回
:::::::::

输出Tensor，与 ``x`` 维度相同、数据类型相同（输入为int时，输出数据类型为float64）。

代码示例
:::::::::

.. code-block:: python

  import paddle

  x = paddle.to_tensor([-1, +1, +1, -1]).astype('float32')
  #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
  #       [-1,  1,  1, -1])

  y = paddle.to_tensor([-1, -1, +1, +1]).astype('float32')
  #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
  #       [-1,  -1,  1, 1])

  out = paddle.atan2(x, y)
  #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
  #       [-2.35619450,  2.35619450,  0.78539819, -0.78539819])