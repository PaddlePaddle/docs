.. _cn_api_fluid_layers_adaptive_pool2d:

adaptive_pool2d
-------------------------------

.. py:function:: paddle.fluid.layers.adaptive_pool2d(input, pool_size, pool_type='max', require_index=False, name=None)

pooling2d操作根据输入 ``input`` ， ``pool_size`` ， ``pool_type`` 参数计算输出。 输入（X）和输出（Out）采用NCHW格式，其中N是批大小batch size，C是通道数，H是feature(特征)的高度，W是feature（特征）的宽度。 参数 ``pool_size`` 由两个元素构成, 这两个元素分别代表高度和宽度。 输出（Out）的H和W维与 ``pool_size`` 大小相同。


对于平均adaptive pool2d:

..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

参数：
  - **input** （Variable） - 池化操作的输入张量。 输入张量的格式为NCHW，其中N是batch大小，C是通道数，H是特征的高度，W是特征的宽度。
  - **pool_size** （int | list | tuple） - 池化核大小。 如果池化核大小是元组或列表，则它必须包含两个整数（pool_size_Height，pool_size_Width）。
  - **pool_type** （string）- 池化类型，可输入“max”代表max-pooling，或者“avg”代表average-pooling。
  - **require_index** （bool） - 如果为true，则输出中带有最大池化点所在的索引。 如果pool_type为avg,该项不可被设置为true。
  - **name** （str | None） - 此层的名称（可选）。 如果设置为None，则将自动命名该层。


返回： 池化结果

返回类型: Variable


抛出异常:

  - ``ValueError`` – ``pool_type`` 不是 ‘max’ 或 ‘avg’
  - ``ValueError`` – 当 ``pool_type`` 是 ‘avg’ 时，错误地设置 ‘require_index’ 为true .
  - ``ValueError`` – ``pool_size`` 应为一个长度为2的列表或元组

.. code-block:: python

    # 假设输入形为[N, C, H, W], `pool_size` 为 [m, n],
    # 输出形为 [N, C, m, n], adaptive pool 将输入的 H 和 W 维度
    # 平均分割为 m * n 个栅格(grid) ，然后为每个栅格进行池化得到输出
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
    data = fluid.layers.data(
        name='data', shape=[3, 32, 32], dtype='float32')
    pool_out = fluid.layers.adaptive_pool2d(
                      input=data,
                      pool_size=[3, 3],
                      pool_type='avg')




