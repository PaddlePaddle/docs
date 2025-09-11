.. _cn_api_paddle_static_nn_fc:

fc
-------------------------------


.. py:function::  paddle.static.nn.fc(x, size, num_flatten_dims=1, weight_attr=None, bias_attr=None, activation=None, name=None)


在神经网络中构建一个全连接层。其输入可以是一个 Tensor 或多个 Tensor 组成的 list（详见参数说明）。为每个输入 Tensor 创建一个权重（weight）参数，即一个从每个输入单元到每个输出单元的全连接权重矩阵。
每个输入 Tensor 和其对应的权重（weight）相乘得到形状为 :math:`[batch\_size, *, size]` 输出 Tensor，其中 :math:`*` 表示可以为任意个额外的维度。
如果有多个输入 Tensor，则多个形状为 :math:`[batch\_size, *, size]` 的 Tensor 计算结果会被累加起来，作为最终输出。如果 :attr:`bias_attr` 非空，则会创建一个偏置（bias）参数，并把它累加到输出 Tensor 中。
如果 :attr:`activation` 非空，将会在输出结果上应用相应的激活函数。

对于单个输入 Tensor :math:`X`，计算公式为：

.. math::

        \\Out = Act({XW + b})\\



对于多个 Tensor，计算公式为：

.. math::

        \\Out=Act(\sum^{M-1}_{i=0}X_iW_i+b) \\


其中：

- :math:`M`：输入 Tensor 的个数。如果输入是 Tensor 列表，:math:`M` 等于 :math:`len(X)`；
- :math:`X_i`：第 i 个输入 Tensor；
- :math:`W_i`：对应第 i 个输入 Tensor 的权重矩阵；
- :math:`b`：偏置参数；
- :math:`Act` ：激活函数；
- :math:`Out`：输出 Tensor。


.. code-block:: text

    # Case 1, input is a single tensor:
    x.data = [[[0.1, 0.2],
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

- **x** (Tensor|list of Tensor) – 由一个多维 Tensor 或由多个 Tensor 组成的 list，每个输入 Tensor 的维度至少是 2。数据类型可以为 float16，float32 或 float64。
- **size** (int) – 全连接层输出单元的数目，即输出 Tensor 的特征维度。
- **num_flatten_dims** (int，可选) – 输入可以接受维度大于 2 的 Tensor。在计算时，输入首先会被扁平化为一个二维矩阵，之后再与权重相乘。参数 :attr:`num_flatten_dims` 决定了输入 Tensor 扁平化的方式：前 :math:`num\_flatten\_dims` (包含边界，从 1 开始数) 个维度会被扁平化为二维矩阵的第一维 (即为矩阵的高)，剩下的 :math:`rank(x) - num\_flatten\_dims` 维被扁平化为二维矩阵的第二维 (即矩阵的宽)。例如，假设 :attr:`x` 是一个五维的 Tensor，其形状为 :math:`[2, 3, 4, 5, 6]` ， :attr:`num_flatten_dims` = 3 时扁平化后的矩阵形状为 :math:`[2 * 3 * 4, 5 * 6] = [24, 30]`，最终输出 Tensor 的形状为 :math:`[2, 3, 4, size]`。默认值为 1。
- **weight_attr** (ParamAttr，可选) – 指定权重参数的属性。默认值为 None，表示使用默认的权重参数属性，将权重参数初始化为 0。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。注意：如果该 api 输入 x 为一个 Tensor 的数组，那 **weight_attr** 也应该是一个同样长度的数组，并且与 x 数组一一对应。
- **bias_attr** (ParamAttr|bool，可选) – 指定偏置参数的属性。:attr:`bias_attr` 为 bool 类型且设置为 False 时，表示不会为该层添加偏置。:attr:`bias_attr` 如果设置为 True 或者 None，则表示使用默认的偏置参数属性，将偏置参数初始化为 0。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。默认值为 None。
- **activation** (str，可选) – 应用于输出上的激活函数，如 tanh、softmax、sigmoid，relu 等，支持列表请参考 :ref:`api_guide_activations`，默认值为 None。
- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::

Tensor，形状为 :math:`[batch\_size, *, size]`，数据类型与输入 Tensor 相同。



代码示例
:::::::::

COPY-FROM: paddle.static.nn.fc
