.. _cn_api_fluid_layers_deformable_roi_pooling:

deformable_roi_pooling
-------------------------------

.. py:function:: paddle.fluid.layers.deformable_roi_pooling(input, rois, trans, no_trans=False, spatial_scale=1.0, group_size=[1, 1], pooled_height=1, pooled_width=1, part_size=None, sample_per_part=1, trans_std=0.1, position_sensitive=False, name=None)




可变形感兴趣区域（ROI）池化层

该OP对输入进行了可形变的感兴趣区域(ROI)池化操作。如同 `可形变卷积网络 <https://arxiv.org/abs/1703.06211>`_ 描述的一样，它将为每个bin中的像素获取一个偏移量，以便于在合适的位置进行池化。在完成可变形感兴趣区域（ROI）池化操作之后，批量数将变为候选框的数量。

可变形感兴趣区域（ROI）池化包含三个步骤：
    
1、将获取的候选区域按照设定的池化宽度和池化高度划分成相同大小的区域。

2、将得到的位置偏移量添加到候选区域的像素来得到新的位置，并且通过双线性插值去获取那些偏移之后位置不为整数的像素的值。

3、在每一个bin中去均匀采样一些像素点，获取其中的均值去作为我们的输出。


参数
::::::::::::

    - **input** (Variable) - 可变形感兴趣区域(ROI)池化层的输入，输入为数据类型为float32的Tensor。输入张量的形状为[N，C，H，W]。其中N是批量大小，C是输入通道的数量，H是特征的高度，W是特征的宽度。
    - **rois** （Variable）- 将池化的ROIs（感兴趣区域），应为一个形状为(num_rois，4)的2-D LoDTensor，且lod level为1。其中值为[[x1，y1，x2，y2]，...]，(x1，y1)为左上角坐标，(x2， y2)为右下角坐标。
    - **trans** （Variable）- 池化时ROIs上的特征偏移，输入为数据类型为float32的Tensor。格式为[N，C，H，W]，其中N是ROIs的数量，C是通道的数量，指示x和y方向上的偏移距离，H是池化的高度，W是池化的宽度。
    - **no_trans** （bool）- 确定roi池化阶段是否加入偏移以获取新的输出。其中值为bool变量，取True或False。如果为True，则表示不加入偏移。默认为False。
    - **spatial_scale** (float) - 输入特征图的高度（或宽度）与原始图像高度（或宽度）的比率，其中数值的类型为float32，并且等于卷积图层中总步长的倒数，默认为1.0。
    - **group_size** （list|tuple）- 输入通道划分成的组数，输入为list 或者 tuple，其中数值类型为int32（例如，输入通道的数量是k1 * k2 * (C + 1)，其中k1和k2是组宽度和高度，C + 1是输出通道的数量。如（4，6）中4是组的高度，6是组的宽度）。默认为[1，1]。
    - **pooled_height** （int）- 池化后输出的高度，值的类型为int32，默认值：1。
    - **pooled_width** （int）- 池化后输出的宽度，值的类型为int32，默认值：1。
    - **part_size** （list|tuple）- 偏移的高度和宽度，如(4，6)代表高度为4、宽度为6，常规是高度和宽度等于pooled_height和pooled_width。默认为None，此时默认值为[pooled_height，pooled_width]。
    - **sample_per_part** （int）- 每个bin中的样本数量，设置值越大，采样结果越精细，但是更加消耗性能。默认为1。
    - **trans_std** （float）- 偏移系数，控制偏移量的大小，默认为0.1。
    - **position_sensitive** （bool）- 是否选择可变形位置敏感型感兴趣区域（PSROI）池化模式，数值类型为bool型。如果为False，输入维度和输出维度相等。如果为True，输入维度等于输出维度乘以pooled_width和pooled_height。默认为False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 可变形感兴趣区域(ROI)池化的输出，如果position_sensitive为False，输出维度和输出维度相等。如果position_sensitive为True，输出维度等于输入维度除以pooled_width和pooled_height。


返回类型
::::::::::::
 Variable，数据类型为float32。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.deformable_roi_pooling