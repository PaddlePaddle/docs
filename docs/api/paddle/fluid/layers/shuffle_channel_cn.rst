.. _cn_api_fluid_layers_shuffle_channel:

shuffle_channel
-------------------------------

.. py:function:: paddle.fluid.layers.shuffle_channel(x, group, name=None)




该OP将输入 ``x`` 的通道混洗重排。它将每个组中的输入通道分成 ``group`` 个子组，并通过逐一从每个子组中选择元素来获得新的顺序。

请参阅 https://arxiv.org/pdf/1707.01083.pdf

::

    输入一个形为 (N, C, H, W) 的4-D tensor:

    input.shape = (1, 4, 2, 2)
    input.data =[[[[0.1, 0.2],
                   [0.2, 0.3]],

                  [[0.3, 0.4],
                   [0.4, 0.5]],

                  [[0.5, 0.6],
                   [0.6, 0.7]],

                  [[0.7, 0.8],
                   [0.8, 0.9]]]]

    指定组数 group: 2
    可得到与输入同形的输出 4-D tensor:

    out.shape = (1, 4, 2, 2)
    out.data = [[[[0.1, 0.2],
                  [0.2, 0.3]],

                 [[0.5, 0.6],
                  [0.6, 0.7]],

                 [[0.3, 0.4],
                  [0.4, 0.5]],

                 [[0.7, 0.8],
                  [0.8, 0.9]]]]

参数
::::::::::::

  - **x** (Variable) – 输入Tensor。维度为[N，C，H，W]的4-D Tensor。
  - **group** (int) – 表示子组的数目，它应该整除通道数。

返回
::::::::::::
一个形状和类型与输入相同的Tensor。

返回类型
::::::::::::
Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.shuffle_channel