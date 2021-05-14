.. _cn_api_nn_functional_bilinear:

bilinear
-------------------------------


.. py:function:: paddle.nn.functional.bilinear(x1, x2, weight, bias=None, name=None)

该层对两个输入执行双线性张量积。
详细的计算和返回值维度请参见 :ref:`cn_api_nn_Bilinear`

参数
:::::::::
  - **x1** (int): 第一个输入的 `Tensor` ，数据类型为：float32、float64。
  - **x2** (int): 第二个输入的 `Tensor` ，数据类型为：float32、float64。
  - **weight** (Parameter) ：本层的可学习参数。形状是 [out_features, in1_features, in2_features]。
  - **bias** (Parameter, 可选) : 本层的可学习偏置。形状是 [1, out_features]。默认值为None，如果被设置成None，则不会有bias加到output结果上。
  - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，一个形为 [batch_size, out_features] 的 2-D 张量。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy
    import paddle.nn.functional as F

    x1 = numpy.random.random((5, 5)).astype('float32')
    x2 = numpy.random.random((5, 4)).astype('float32')
    w = numpy.random.random((1000, 5, 4)).astype('float32')
    b = numpy.random.random((1, 1000)).astype('float32')

    result = F.bilinear(paddle.to_tensor(x1), paddle.to_tensor(x2), paddle.to_tensor(w), paddle.to_tensor(b))           # result shape [5, 1000]


