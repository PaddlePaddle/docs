.. _cn_api_paddle_static_nn_common_fc:

fc
-------------------------------


.. py:function::  paddle.static.nn.fc(x, size, num_flatten_dims=1, weight_attr=None, bias_attr=None, activation=None, name=None)


该OP将在神经网络中构建一个全连接层。其输入可以是一个Tensor或多个Tensor组成的list（详见参数说明）。该OP会为每个输入Tensor创建一个权重（weight）参数，即一个从每个输入单元到每个输出单元的全连接权重矩阵。
每个输入Tensor和其对应的权重（weight）相乘得到形状为 :math:`[batch\_size, *, size]` 输出Tensor，其中 :math:`*` 表示可以为任意个额外的维度。
如果有多个输入Tensor，则多个形状为 :math:`[batch\_size, *, size]` 的Tensor计算结果会被累加起来，作为最终输出。如果 :attr:`bias_attr` 非空，则会创建一个偏置（bias）参数，并把它累加到输出Tensor中。
如果 :attr:`activation` 非空，将会在输出结果上应用相应的激活函数。

对于单个输入Tensor ::math`X` ，计算公式为：

.. math::

        \\Out = Act({XW + b})\\



对于多个Tensor，计算公式为：

.. math::

        \\Out=Act(\sum^{M-1}_{i=0}X_iW_i+b) \\


其中：

- :math:`M` ：输入Tensor的个数。如果输入是Tensor列表，:math:`M` 等于 :math:`len(X)`
- :math:`X_i` ：第i个输入Tensor
- :math:`W_i` ：对应第i个输入Tensor的权重矩阵
- :math:`b` ：偏置参数
- :math:`Act` ：activation function (激活函数)
- :math:`Out` ：输出Tensor

           
.. code-block:: text

    # Case 1, input is a single tensor:
    data = [[[0.1, 0.2],
             [0.3, 0.4]]]
    x.shape = (1, 2, 2) # 1 is batch_size

    out = paddle.static.nn.fc(x=x, size=1, num_flatten_dims=2)

    # Get the output:
    out.data = [[0.83234344], [0.34936576]]
    out.shape = (1, 2, 1)

    # Case 2, input is a list of tensor:
    x0.data = [[[0.1, 0.2],
                [0.3, 0.4]]]
    x0.shape = (1, 2, 2) # 1 is batch_size

    x1.data = [[[0.1, 0.2, 0.3]]]
    x1.shape = (1, 1, 3)

    out = paddle.static.nn.fc(x=[x0, x1], size=2)

    # Get the output:
    out.data = [[0.18669507, 0.1893476]]
    out.shape = (1, 2)


参数
:::::::::

- **x** (Tensor|list of Tensor) – 一个多维Tensor或由多个Tensor组成的list，每个输入Tensor的维度至少是2。数据类型可以为float16，float32或float64。
- **size** (int) – 全连接层输出单元的数目，即输出Tensor的特征维度。
- **num_flatten_dims** (int) – 输入可以接受维度大于2的Tensor。在计算时，输入首先会被扁平化为一个二维矩阵，之后再与权重相乘。参数 :attr:`num_flatten_dims` 决定了输入Tensor扁平化的方式: 前 :math:`num\_flatten\_dims` (包含边界，从1开始数) 个维度会被扁平化为二维矩阵的第一维 (即为矩阵的高), 剩下的 :math:`rank(x) - num\_flatten\_dims` 维被扁平化为二维矩阵的第二维 (即矩阵的宽)。 例如， 假设 :attr:`x` 是一个五维的Tensor，其形状为 :math:`[2, 3, 4, 5, 6]` ， :attr:`num_flatten_dims` = 3时扁平化后的矩阵形状为 :math:`[2 * 3 * 4, 5 * 6] = [24, 30]` ，最终输出Tensor的形状为 :math:`[2, 3, 4, size]` 。默认值为1。
- **weight_attr** (ParamAttr, 可选) – 指定权重参数的属性。默认值为None，表示使用默认的权重参数属性，将权重参数初始化为0。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
- **bias_attr** (ParamAttr|bool, 可选) – 指定偏置参数的属性。:attr:`bias_attr` 为bool类型且设置为False时，表示不会为该层添加偏置。 :attr:`bias_attr` 如果设置为True或者None，则表示使用默认的偏置参数属性，将偏置参数初始化为0。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。默认值为None。
- **activation** (str, 可选) – 应用于输出上的激活函数，如tanh、softmax、sigmoid，relu等，支持列表请参考 :ref:`api_guide_activations` ，默认值为None。
- **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。


返回
:::::::::

Tensor，形状为 :math:`[batch\_size, *, size]` ，数据类型与输入Tensor相同。


抛出异常
:::::::::

- :math:`ValueError` - 如果输入Tensor的维度小于2


代码示例
:::::::::


.. code-block:: python

    import paddle
    paddle.enable_static()

    # When input is a single tensor
    x = paddle.static.data(name="x", shape=[1, 2, 2], dtype="float32")
    # x: [[[0.1 0.2]
    #      [0.3 0.4]]]
    out = paddle.static.nn.fc(
        x=x,
        size=1,
        num_flatten_dims=2,
        weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)),
        bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)))
    # out: [[[1.15]
    #        [1.35]]]

    # When input is multiple tensors
    x0 = paddle.static.data(name="x0", shape=[1, 2, 2], dtype="float32")
    # x0: [[[0.1 0.2]
    #       [0.3 0.4]]]
    x1 = paddle.static.data(name="x1", shape=[1, 1, 3], dtype="float32")
    # x1: [[[0.1 0.2 0.3]]]
    out = paddle.static.nn.fc(
        x=[x0, x1],
        size=2,
        weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)),
        bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)))
    # out: [[1.8 1.8]]


