.. _cn_api_fluid_layers_adaptive_pool2d:

adaptive_pool2d
-------------------------------

.. py:function:: paddle.fluid.layers.adaptive_pool2d(input, pool_size, pool_type='max', require_index=False, name=None)




该 OP 使用上述输入参数的池化配置，为二维空间自适应池化操作，根据 ``input``，池化类型 ``pool_type``，池化核大小 ``pool_size`` 这些参数得到输出。

输入 X 和输出 Out 是 NCHW 格式，N 为批大小，C 是通道数，H 是特征高度，W 是特征宽度。参数 ``pool_size`` 含有两个整型元素，分别代表高度和宽度上的参数。输出 Out 的 H 和 W 维由 ``pool_size`` 决定，即输出 shape 为 :math:`\left ( N,C,pool_size[0],pool_size[1] \right )`


对于平均 adaptive pool2d:

..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

参数
::::::::::::

  - **input** （Variable） - 池化操作的输入 Tensor，维度为 :math:`[N, C, H, W]` 的 4-D Tensor。输入 Tensor 的格式为 NCHW，其中 N 是 batch 大小，C 是通道数，H 是特征的高度，W 是特征的宽度，数据类型为 float32 或 float64。
  - **pool_size** （int|list|tuple） - 池化核大小。如果池化核大小是元组或列表，则它必须包含两个整数（pool_size_Height，pool_size_Width。若为一个整数，则表示 H 和 W 维度上均为该值。
  - **pool_type** （string）- 池化类型，可输入“max”代表 max-pooling，或者“avg”代表 average-pooling。
  - **require_index** （bool） - 如果为 False，则输出中带有最大池化点所在的索引。如果 pool_type 为 avg，该项不可被设置为 True，默认 False。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
 Variable(Tensor) 自适应池化结果 Tensor

返回类型
::::::::::::
变量(Variable)，数据类型与 ``input`` 一致


抛出异常
::::::::::::


  - ``ValueError`` – ``pool_type`` 不是 ‘max’ 或 ‘avg’
  - ``ValueError`` – 当 ``pool_type`` 是 ‘avg’ 时，错误地设置 ‘require_index’ 为 true 。
  - ``ValueError`` – ``pool_size`` 应为一个长度为 2 的列表或元组

.. code-block:: python

    # average adaptive pool2d
    # 假设输入形为[N, C, H, W], `pool_size` 为 [m, n],
    # 输出形为 [N, C, m, n], adaptive pool 将输入的 H 和 W 维度
    # 平均分割为 m * n 个栅格(grid)，然后为每个栅格进行池化得到输出
    # adaptive average pool 进行如下操作
    #
    #     for i in range(m):
    #         for j in range(n):
    #             hstart = floor(i * H / m)
    #             hend = ceil((i + 1) * H / m)
    #             wstart = floor(i * W / n)
    #             wend = ceil((i + 1) * W / n)
    #             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
    #
    import paddle.fluid as fluid
    data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
    pool_out = fluid.layers.adaptive_pool2d(
                      input=data,
                      pool_size=[3, 3],
                      pool_type='avg')

    # max adaptive pool2d
    # 假设输入形为[N, C, H, W], `pool_size` 为 [m, n],
    # 输出形为 [N, C, m, n], adaptive pool 将输入的 H 和 W 维度
    # 平均分割为 m * n 个栅格(grid)，然后为每个栅格进行池化得到输出
    # adaptive average pool 进行如下操作
    #
    #     for i in range(m):
    #         for j in range(n):
    #             hstart = floor(i * H / m)
    #             hend = ceil((i + 1) * H / m)
    #             wstart = floor(i * W / n)
    #             wend = ceil((i + 1) * W / n)
    #             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
    #
    import paddle.fluid as fluid
    data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
    pool_out = fluid.layers.adaptive_pool2d(
                      input=data,
                      pool_size=[3, 3],
                      pool_type='max')
