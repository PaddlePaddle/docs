.. _cn_api_fluid_layers_adaptive_pool3d:

adaptive_pool3d
-------------------------------

.. py:function:: paddle.fluid.layers.adaptive_pool3d(input, pool_size, pool_type='max', require_index=False, name=None)




该OP使用上述输入参数的池化配置，为二维空间自适应池化操作，根据 ``input``，池化类型 ``pool_type``，池化核大小 ``pool_size`` 这些参数得到输出。

输入X和输出Out是NCDHW格式，N为批大小，D是特征深度，C是通道数，H是特征高度，W是特征宽度。参数 ``pool_size`` 含有两个整型元素，分别代表深度，高度和宽度上的参数。输出Out的D, H和W维由 ``pool_size`` 决定，即输出shape为 :math:`\left ( N,C,pool_size[0],pool_size[1],pool_size[2] \right )`


对于平均adaptive pool3d:

..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}



参数
::::::::::::

  - **input** （Variable） - 池化操作的输入张量，维度为 :math:`[N, C, D, H, W]` 的5-D Tensor。输入张量的格式为NCDHW，其中N是batch大小，C是通道数，D为特征的深度，H是特征的高度，W是特征的宽度，数据类型为float32或float64。
  - **pool_size** （int|list|tuple） - 池化核大小。如果池化核大小是元组或列表，则它必须包含三个整数（Depth, Height, Width）。若为一个整数，则表示D, H和W维度上均为该值。
  - **pool_type** （string）- 池化类型，可输入“max”代表max-pooling，或者“avg”代表average-pooling。
  - **require_index** （bool） - 如果为True，则输出中带有最大池化点所在的索引。如果pool_type为avg，该项不可被设置为True，默认False。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
 Variable(Tensor) 自适应池化结果张量

返回类型
::::::::::::
变量(Variable)，数据类型与 ``input`` 一致


抛出异常
::::::::::::


  - ``ValueError`` – ``pool_type`` 不是 ‘max’ 或 ‘avg’
  - ``ValueError`` – 当 ``pool_type`` 是 ‘avg’ 时，错误地设置 ‘require_index’ 为true 。
  - ``ValueError`` – ``pool_size`` 应为一个长度为3的列表或元组

.. code-block:: python

    # average adaptive pool2d
    # 假设输入形为[N, C, D, H, W], `pool_size` 为 [l, m, n],
    # 输出形为 [N, C, l, m, n], adaptive pool 将输入的D, H 和 W 维度
    # 平均分割为 l * m * n 个栅格(grid)，然后为每个栅格进行池化得到输出
    # adaptive average pool 进行如下操作
    #
    #     for i in range(l):
    #         for j in range(m):
    #             for k in range(n):
    #                 dstart = floor(i * D / l)
    #                 dend = ceil((i + 1) * D / l)
    #                 hstart = floor(j * H / m)
    #                 hend = ceil((j + 1) * H / m)
    #                 wstart = floor(k * W / n)
    #                 wend = ceil((k + 1) * W / n)
    #                 output[:, :, i, j, k] =
    #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
    #
    
    import paddle.fluid as fluid

    data = fluid.data(name='data', shape=[None, 3, 32, 32, 32], dtype='float32')
    pool_out = fluid.layers.adaptive_pool3d(
                      input=data,
                      pool_size=[3, 3, 3],
                      pool_type='avg')

    # max adaptive pool2d
    # 假设输入形为[N, C, D, H, W], `pool_size` 为 [l, m, n],
    # 输出形为 [N, C, l, m, n], adaptive pool 将输入的D, H 和 W 维度
    # 平均分割为 l * m * n 个栅格(grid)，然后为每个栅格进行池化得到输出
    # adaptive average pool 进行如下操作
    #
    #     for i in range(l):
    #         for j in range(m):
    #             for k in range(n):
    #                 dstart = floor(i * D / l)
    #                 dend = ceil((i + 1) * D / l)
    #                 hstart = floor(j * H / m)
    #                 hend = ceil((j + 1) * H / m)
    #                 wstart = floor(k * W / n)
    #                 wend = ceil((k + 1) * W / n)
    #                 output[:, :, i, j, k] =
    #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
    #
    
    import paddle.fluid as fluid

    data = fluid.data(name='data', shape=[None, 3, 32, 32, 32], dtype='float32')
    pool_out = fluid.layers.adaptive_pool3d(
                      input=data,
                      pool_size=[3, 3, 3],
                      pool_type='max')




