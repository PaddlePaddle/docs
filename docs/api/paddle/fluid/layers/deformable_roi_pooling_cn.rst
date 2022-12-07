.. _cn_api_fluid_layers_deformable_roi_pooling:

deformable_roi_pooling
-------------------------------

.. py:function:: paddle.fluid.layers.deformable_roi_pooling(input, rois, trans, no_trans=False, spatial_scale=1.0, group_size=[1, 1], pooled_height=1, pooled_width=1, part_size=None, sample_per_part=1, trans_std=0.1, position_sensitive=False, name=None)




可变形感兴趣区域（ROI）池化层

该 OP 对输入进行了可形变的感兴趣区域(ROI)池化操作。如同 `可形变卷积网络 <https://arxiv.org/abs/1703.06211>`_ 描述的一样，它将为每个 bin 中的像素获取一个偏移量，以便于在合适的位置进行池化。在完成可变形感兴趣区域（ROI）池化操作之后，批量数将变为候选框的数量。

可变形感兴趣区域（ROI）池化包含三个步骤：

1、将获取的候选区域按照设定的池化宽度和池化高度划分成相同大小的区域。

2、将得到的位置偏移量添加到候选区域的像素来得到新的位置，并且通过双线性插值去获取那些偏移之后位置不为整数的像素的值。

3、在每一个 bin 中去均匀采样一些像素点，获取其中的均值去作为我们的输出。


参数
::::::::::::

    - **input** (Variable) - 可变形感兴趣区域(ROI)池化层的输入，输入为数据类型为 float32 的 Tensor。输入 Tensor 的形状为[N，C，H，W]。其中 N 是批量大小，C 是输入通道的数量，H 是特征的高度，W 是特征的宽度。
    - **rois** （Variable）- 将池化的 ROIs（感兴趣区域），应为一个形状为(num_rois，4)的 2-D LoDTensor，且 lod level 为 1。其中值为[[x1，y1，x2，y2]，...]，(x1，y1)为左上角坐标，(x2， y2)为右下角坐标。
    - **trans** （Variable）- 池化时 ROIs 上的特征偏移，输入为数据类型为 float32 的 Tensor。格式为[N，C，H，W]，其中 N 是 ROIs 的数量，C 是通道的数量，指示 x 和 y 方向上的偏移距离，H 是池化的高度，W 是池化的宽度。
    - **no_trans** （bool）- 确定 roi 池化阶段是否加入偏移以获取新的输出。其中值为 bool 变量，取 True 或 False。如果为 True，则表示不加入偏移。默认为 False。
    - **spatial_scale** (float) - 输入特征图的高度（或宽度）与原始图像高度（或宽度）的比率，其中数值的类型为 float32，并且等于卷积图层中总步长的倒数，默认为 1.0。
    - **group_size** （list|tuple）- 输入通道划分成的组数，输入为 list 或者 tuple，其中数值类型为 int32（例如，输入通道的数量是 k1 * k2 * (C + 1)，其中 k1 和 k2 是组宽度和高度，C + 1 是输出通道的数量。如（4，6）中 4 是组的高度，6 是组的宽度）。默认为[1，1]。
    - **pooled_height** （int）- 池化后输出的高度，值的类型为 int32，默认值：1。
    - **pooled_width** （int）- 池化后输出的宽度，值的类型为 int32，默认值：1。
    - **part_size** （list|tuple）- 偏移的高度和宽度，如(4，6)代表高度为 4、宽度为 6，常规是高度和宽度等于 pooled_height 和 pooled_width。默认为 None，此时默认值为[pooled_height，pooled_width]。
    - **sample_per_part** （int）- 每个 bin 中的样本数量，设置值越大，采样结果越精细，但是更加消耗性能。默认为 1。
    - **trans_std** （float）- 偏移系数，控制偏移量的大小，默认为 0.1。
    - **position_sensitive** （bool）- 是否选择可变形位置敏感型感兴趣区域（PSROI）池化模式，数值类型为 bool 型。如果为 False，输入维度和输出维度相等。如果为 True，输入维度等于输出维度乘以 pooled_width 和 pooled_height。默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 可变形感兴趣区域(ROI)池化的输出，如果 position_sensitive 为 False，输出维度和输出维度相等。如果 position_sensitive 为 True，输出维度等于输入维度除以 pooled_width 和 pooled_height。


返回类型
::::::::::::
 Variable，数据类型为 float32。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.deformable_roi_pooling
