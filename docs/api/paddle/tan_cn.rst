.. _cn_api_fluid_layers_tan:

tan
-------------------------------

.. py:function:: paddle.tan(x, name=None)
三角函数tangent。

输入范围是 `(k*pi-pi/2, k*pi+pi/2)`， 输出范围是 `[-inf, inf]` 。 

.. math::
    out = tan(x)

参数
:::::::::

  - **x** (Tensor) – 该OP的输入为Tensor。数据类型为float32，float64。 
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。


返回
:::::::::

Tensor - 该OP的输出为Tensor，数据类型为输入一致。


代码示例
:::::::::

..  code-block:: python
  
    import paddle
    # example 1: x is a float
    x_i = paddle.to_tensor([[1.0], [2.0]])
    res = paddle.tan(x_i) #[[ 1.55740786], [-2.18504000]]
    # example 2: x is float32
    x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
    paddle.to_tensor(x_i)
    res = paddle.tan(x_i)
    print(res) # [ 1.55740786]
  
    # example 3: x is float64
    x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
    paddle.to_tensor(x_i)
    res = paddle.tan(x_i)
    print(res) # [ 1.55740786]
