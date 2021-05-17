.. _cn_api_paddle_nn_layer_common_Linear:

Linear
-------------------------------

.. py:class:: paddle.nn.Linear(in_features, out_features, weight_attr=None, bias_attr=None, name=None)


**线性变换层** 。对于每个输入Tensor :math:`X` ，计算公式为：

.. math::

    Out = XW + b

其中， :math:`W` 和 :math:`b` 分别为权重和偏置。

Linear层只接受一个Tensor作为输入，形状为 :math:`[batch\_size, *, in\_features]` ，其中 :math:`*` 表示可以为任意个额外的维度。
该层可以计算输入Tensor与权重矩阵 :math:`W` 的乘积，然后生成形状为 :math:`[batch\_size, *, out\_features]` 的输出Tensor。
如果 :math:`bias\_attr` 不是False，则将创建一个偏置参数并将其添加到输出中。

参数
:::::::::

- **in_features** (int) – 线性变换层输入单元的数目。
- **out_features** (int) – 线性变换层输出单元的数目。
- **weight_attr** (ParamAttr, 可选) – 指定权重参数的属性。默认值为None，表示使用默认的权重参数属性，将权重参数初始化为0。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
- **bias_attr** (ParamAttr|bool, 可选) – 指定偏置参数的属性。 :math:`bias\_attr` 为bool类型且设置为False时，表示不会为该层添加偏置。 :math:`bias\_attr` 如果设置为True或者None，则表示使用默认的偏置参数属性，将偏置参数初始化为0。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。默认值为None。
- **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

属性
:::::::::

.. py:attribute:: weight

本层的可学习参数，类型为 ``Parameter`` 。

.. py:attribute:: bias

本层的可学习偏置，类型为 ``Parameter`` 。

形状
:::::::::

- 输入：形状为 :math:`[batch\_size, *, in\_features]` 的多维Tensor。
- 输出：形状为 :math:`[batch\_size, *, out\_features]` 的多维Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle

    # Define the linear layer.
    weight_attr = paddle.ParamAttr(
        name="weight",
        initializer=paddle.nn.initializer.Constant(value=0.5))
    bias_attr = paddle.ParamAttr(
        name="bias",
        initializer=paddle.nn.initializer.Constant(value=1.0))
    linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
    # linear.weight: [[0.5 0.5 0.5 0.5]
    #                 [0.5 0.5 0.5 0.5]]
    # linear.bias: [1. 1. 1. 1.]

    x = paddle.randn((3, 2), dtype="float32")
    # x: [[-0.32342386 -1.200079  ]
    #     [ 0.7979031  -0.90978354]
    #     [ 0.40597573  1.8095392 ]]
    y = linear(x)
    # y: [[0.23824859 0.23824859 0.23824859 0.23824859]
    #     [0.9440598  0.9440598  0.9440598  0.9440598 ]
    #     [2.1077576  2.1077576  2.1077576  2.1077576 ]]

