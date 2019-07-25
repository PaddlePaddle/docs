.. _cn_api_fluid_layers_adaptive_pool3d:

adaptive_pool3d
-------------------------------

.. py:function:: paddle.fluid.layers.adaptive_pool3d(input, pool_size, pool_type='max', require_index=False, name=None)

pooling3d操作根据输入 ``input`` ，``pool_size`` ， ``pool_type`` 参数计算输出。 输入（X）和输出（输出）采用NCDHW格式，其中N是批大小batch size，C是通道数，D是特征(feature)的深度，H是特征的高度，W是特征的宽度。 参数 ``pool_size`` 由三个元素组成。 这三个元素分别代表深度，高度和宽度。输出（Out）的D,H,W维与 ``pool_size`` 相同。


对于平均adaptive pool3d:

..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}



参数：
  - **input** （Variable） - 池化操作的输入张量。 输入张量的格式为NCDHW，其中N是batch大小，C是通道数，D为特征的深度，H是特征的高度，W是特征的宽度。
  - **pool_size** （int | list | tuple） - 池化核大小。 如果池化核大小是元组或列表，则它必须包含三个整数（Depth, Height, Width）。
  - **pool_type** （string）- 池化类型，可输入“max”代表max-pooling，或者“avg”代表average-pooling。
  - **require_index** （bool） - 如果为true，则输出中带有最大池化点所在的索引。 如果pool_type为avg,该项不可被设置为true。
  - **name** （str | None） - 此层的名称（可选）。 如果设置为None，则将自动命名该层。


返回： 池化结果

返回类型: Variable


抛出异常:

  - ``ValueError`` – ``pool_type`` 不是 ‘max’ 或 ‘avg’
  - ``ValueError`` – 当 ``pool_type`` 是 ‘avg’ 时，错误地设置 ‘require_index’ 为true .
  - ``ValueError`` – ``pool_size`` 应为一个长度为3的列表或元组

.. code-block:: python

    # 假设输入形为[N, C, D, H, W], `pool_size` 为 [l, m, n],
    # 输出形为 [N, C, l, m, n], adaptive pool 将输入的D, H 和 W 维度
    # 平均分割为 l * m * n 个栅格(grid) ，然后为每个栅格进行池化得到输出
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

    data = fluid.layers.data(
    name='data', shape=[3, 32, 32, 32], dtype='float32')
    pool_out, mask = fluid.layers.adaptive_pool3d(
                      input=data,
                      pool_size=[3, 3, 3],
                      pool_type='avg')




