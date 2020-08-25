.. _cn_api_nn_Bilinear:

Bilinear
-------------------------------

.. py:function:: paddle.nn.Bilinear(in1_features, in2_features, out_features, weight_attr=None, bias_attr=None, name=None)

该层对两个输入执行双线性张量积。

例如:

.. math::

       out_{i} = x1 * W_{i} * {x2^\mathrm{T}}, i=0,1,...,size-1

       out = out + b

在这个公式中：
  - :math:`x1`: 第一个输入，包含 :in1_features个元素，形状为 [batch_size, in1_features]。
  - :math:`x2`: 第二个输入，包含 :in2_features个元素，形状为 [batch_size, in2_features]。
  - :math:`W_{i}`: 第 :i个被学习的权重，形状是 [in1_features, in2_features]。
  - :math:`out_{i}`: 输出的第 :i个元素，形状是 [batch_size, out_features]。
  - :math:`b`: 被学习的偏置参数，形状是 [1, out_features]。
  - :math:`x2^\mathrm{T}`: :math:`x2` 的转置。

参数
:::::::::
  - **in1_features** (int): 每个 **x1** 元素的维度。
  - **in2_features** (int): 每个 **x2** 元素的维度。
  - **out_features** (int): 输出张量的维度。
  - **weight_attr** (ParamAttr，可选) ：指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。
  - **bias_attr** (ParamAttr，可选) : 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性，此时bias的元素会被初始化成0。如果设置成False，则不会有bias加到output结果上。
  - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

属性
:::::::::
    - **weight** 本层的可学习参数，类型为 Parameter
    - **bias** 本层的可学习偏置，类型为 Parameter

返回
:::::::::
``Tensor``，一个形为 [batch_size, out_features] 的 2-D 张量。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy

    paddle.disable_static()
    layer1 = numpy.random.random((5, 5)).astype('float32')
    layer2 = numpy.random.random((5, 4)).astype('float32')
    bilinear = paddle.nn.Bilinear(
        in1_features=5, in2_features=4, out_features=1000)
    result = bilinear(paddle.to_tensor(layer1),
                    paddle.to_tensor(layer2))     # result shape [5, 1000]

