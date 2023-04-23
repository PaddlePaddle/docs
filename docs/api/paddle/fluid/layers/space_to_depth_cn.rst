.. _cn_api_fluid_layers_space_to_depth:

space_to_depth
-------------------------------

.. py:function:: paddle.fluid.layers.space_to_depth(x, blocksize, name=None)




该 OP 对成块的空间数据进行重组，输出一个输入 Tensor 的拷贝，其高度和宽度维度上的值移至通道维度。

重组时，依据 ``blocksize`` 指明的数据块大小，对形为 ``[batch, channel, height, width]`` 的输入 Tensor 进行 space_to_depth（广度至深度）运算，生成形为 ``[batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]``  的输出：

 - 在各位置上，不重叠的，大小为 ``blocksize * blocksize`` 的块重组入深度 depth
 - 输入各个块中的 Y, X 坐标变为输出 Tensor 通道索引的高序部位
 - 输入 ``channel`` 需可以被 ``blocksize`` 的平方整除
 - 输入的高度和宽度需可以被 ``blocksize`` 整除

该 OP 适用于在卷积间重放缩激活函数，并保持所有的数据。

范例如下：

::

    给定形状为[1, 1, 4, 4]的输入 x：
      x.data = [[[[1,   2,  5,  6],
                  [3,   4,  7,  8],
                  [9,  10, 13, 14],
                  [11, 12, 15, 16]]]]
    设置 blocksize = 2

    得到形状为[1, 4, 2, 2]的输出 out：
      out.data = [[[[1,   2],  [3,  4]],
                   [[5,   6],  [7,  8]],
                   [[9,  10], [11, 12]],
                   [[13, 14], [15, 16]]]]




参数
::::::::::::

  - **x** (Variable) – 输入，形状为 ``[batch, channel, height, width]`` 的 4 维 Tensor 或 LoD Tensor。数据类型支持 int32，int64，float32 或 float64。
  - **blocksize** (int) – 在每个特征图上选择元素时采用的块大小，应该 >= 2
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出，形状为 ``[batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]``  的 4 维 Tensor 或 LoD Tensor。数据类型与输入 ``x`` 一致。

返回类型
::::::::::::
Variable

抛出异常
::::::::::::

  - ``TypeError`` - ``blocksize`` 必须是 int64 类型

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.space_to_depth
