.. _cn_api_fluid_layers_row_conv:

row_conv
-------------------------------

.. py:function:: paddle.fluid.layers.row_conv(input, future_context_size, param_attr=None, act=None)

行卷积（Row-convolution operator）称为超前卷积（lookahead convolution）。下面关于DeepSpeech2的paper中介绍了这个operator

    `<http://www.cs.cmu.edu/~dyogatam/papers/wang+etal.iclrworkshop2016.pdf>`_

双向的RNN在深度语音模型中很有用，它通过对整个序列执行正向和反向传递来学习序列的表示。然而，与单向RNNs不同的是，在线部署和低延迟设置中，双向RNNs具有难度。超前卷积将来自未来子序列的信息以一种高效的方式进行计算，以改进单向递归神经网络。 row convolution operator 与一维序列卷积不同，计算方法如下:

给定输入序列长度为 :math:`t` 的输入序列 :math:`X` 和输入维度 :math:`D` ，以及一个大小为 :math:`context * D` 的滤波器 :math:`W` ，输出序列卷积为:

.. math::
    out_i = \sum_{j=i}^{i+context-1} X_{j} · W_{j-i}

公式中：
    - :math:`out_i` : 第i行输出变量形为[1, D].
    - :math:`context` ： 下文（future context）大小
    - :math:`X_j` : 第j行输出变量,形为[1，D]
    - :math:`W_{j-i}` : 第(j-i)行参数，其形状为[1,D]。

详细请参考 `设计文档  <https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645>`_  。

参数:
    - **input** (Variable) -- 输入是一个LodTensor，它支持可变时间长度的输入序列。这个LodTensor的内部张量是一个具有形状(T x N)的矩阵，其中T是这个mini batch中的总的timestep，N是输入数据维数。
    - **future_context_size** (int) -- 下文大小。请注意，卷积核的shape是[future_context_size + 1, D]。
    - **param_attr** (ParamAttr) --  参数的属性，包括名称、初始化器等。
    - **act** (str) -- 非线性激活函数。

返回: 输出(Out)是一个LodTensor，它支持可变时间长度的输入序列。这个LodTensor的内部量是一个形状为 T x N 的矩阵，和X的 shape 一样。


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid

  x = fluid.layers.data(name='x', shape=[16],
                        dtype='float32', lod_level=1)
  out = fluid.layers.row_conv(input=x, future_context_size=2)


