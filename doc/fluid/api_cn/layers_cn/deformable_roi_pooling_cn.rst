.. _cn_api_fluid_layers_deformable_roi_pooling:

deformable_roi_pooling
-------------------------------

.. py:function:: paddle.fluid.layers.deformable_roi_pooling(input, rois, trans, no_trans=False, spatial_scale=1.0, group_size=[1, 1], pooled_height=1, pooled_width=1, part_size=None, sample_per_part=1, trans_std=0.1, position_sensitive=False, name=None)

可变形PSROI池层

参数:
    - **input** (Variable) - 可变形PSROI池层的输入。输入张量的形状为[N，C，H，W]。其中N是批量大小，C是输入通道的数量，H是特征的高度，W是特征的宽度。
    - **rois** （Variable）- 将池化的ROIs（感兴趣区域）。应为一个形状为(num_rois, 4)的2-D LoDTensor，且lod level为1。给出[[x1, y1, x2, y2], ...]，(x1, y1)为左上角坐标，(x2, y2)为右下角坐标。
    - **trans** （Variable）- 池化时ROIs上的特征偏移。格式为NCHW，其中N是ROIs的数量，C是通道的数量，指示x和y方向上的偏移距离，H是池化的高度，W是池化的宽度。
    - **no_trans** （bool）- roi池化阶段是否加入偏移以获取新值。取True或False。默认为False。
    - **spatial_scale** (float) - 输入特征图的高度（或宽度）与原始图像高度（或宽度）的比率。等于卷积图层中总步长的倒数，默认为1.0。
    - **group_size** （list|tuple）- 输入通道划分成的组数（例如，输入通道的数量是k1 * k2 *（C + 1），其中k1和k2是组宽度和高度，C + 1是输出通道的数量。如（ 4,6）中4是组的高度，6是组的宽度）。默认为[1,1]。
    - **pooled_height** （integer）- 池化后输出的高度。
    - **pooled_width** （integer）- 池化后输出的宽度。
    - **part_size** （list|tuple）- 偏移高度和宽度，如(4, 6)代表高度为4、宽度为6，默认为None，此时默认值[pooled_height, pooled_width]。
    - **sample_per_part** （integer）- 每个bin中的样本数量，默认为1。
    - **trans_std** （float）- 偏移系数，默认为0.1。
    - **position_sensitive** （bool）- 是否选择可变形psroi池化模式，默认为False。
    - **name** （str）- 层名，默认为None。

返回: 存储可变形psroi池层的张量变量

返回类型:  变量(Variable)

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input",
                              shape=[2, 192, 64, 64],
                              dtype='float32',
                              append_batch_size=False)
    rois = fluid.layers.data(name="rois",
                             shape=[4],
                             dtype='float32',
                             lod_level=1)
    trans = fluid.layers.data(name="trans",
                              shape=[2, 384, 64, 64],
                              dtype='float32',
                              append_batch_size=False)
    x = fluid.layers.nn.deformable_roi_pooling(input=input,
                                                 rois=rois,
                                                 trans=trans,
                                                 no_trans=False,
                                                 spatial_scale=1.0,
                                                 group_size=(1, 1),
                                                 pooled_height=8,
                                                 pooled_width=8,
                                                 part_size=(8, 8),
                                                 sample_per_part=4,
                                                 trans_std=0.1,
                                                 position_sensitive=False)

