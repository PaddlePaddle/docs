.. _cn_api_paddle_nn_layer_Linear:

Linear
-------------------------------

.. py:class:: paddle.nn.layer.common.Linear(in_features, out_features, weight_attr=None, bias_attr=None, name=None)


**线性变换层** 。对于每个输入Tensor X，计算公式为：

.. math::

    \\Out = X * W + b\\

其中， :math:`W` 和 :math:`b` 分别为权重和偏置。

Linear 层只接受一个 Tensor 作为输入，形状为 :math:`[N, *, in_features]` ，其中 N 是 batch_size， :math:`*` 表示可以为任意个额外的维度。
该层可以计算输入 Tensor 与权重矩阵 :math:`W` 的乘积，然后生成形状为 :math:`[N，*，output_features]` 的输出张量。
如果 bias_attr 不是 None，则将创建一个 bias 变量并将其添加到输出中。

参数:
- **in_features** (int) – 线性变换层输入单元的数目。
- **out_features** (int) – 线性变换层输出单元的数目。
- **param_attr** (ParamAttr, 可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
- **bias_attr** (ParamAttr, 可选) – 指定偏置参数属性的对象，若 `bias_attr` 为bool类型，如果设置为False，表示不会为该层添加偏置；如果设置为True，表示使用默认的偏置参数属性。默认值为None，表示使用默认的偏置参数属性。默认的偏置参数属性将偏置参数的初始值设为0。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
- **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回：形状为 :math:`[N，*，output_features]` 的结果张量。

**代码示例**

..  code-block:: python

    import paddle

    # Define the linear layer.
    weight_attr = paddle.ParamAttr(name="weight", initializer=paddle.fluid.initializer.ConstantInitializer(value=0.5))
    bias_attr = paddle.ParamAttr(name="bias", initializer=paddle.fluid.initializer.ConstantInitializer(value=1.0))
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


属性
::::::::::::
.. py:attribute:: weight

本层的可学习参数，类型为 ``Parameter`` 。

.. py:attribute:: bias

本层的可学习偏置，类型为 ``Parameter`` 。

