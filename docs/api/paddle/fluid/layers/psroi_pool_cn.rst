.. _cn_api_fluid_layers_psroi_pool:

psroi_pool
-------------------------------

.. py:function:: paddle.fluid.layers.psroi_pool(input, rois, output_channels, spatial_scale, pooled_height, pooled_width, name=None)




**注意 rois 必须为 2 维 LoDTensor，lod_level 为 1**

该 OP 执行 PSROIPooling 运算，是位置敏感的感兴趣区域池化方法（Position sensitive region of interest pooling，也称为 PSROIPooling）。输入 input 是位置敏感的评分图，输入 rois 是感兴趣区域的位置坐标。PSROIPooling 不同于普通 ROIPooling 的地方在于，输入 input 特征图的不同通道会跟输出特征图上的位置区域相关联，该方法是在 R-FCN 模型中首次提出来的，更多详细信息请参阅 https://arxiv.org/abs/1605.06409。


**样例**：

::

      Given:
        input.shape = [2, 490, 28, 28]
        rois.shape = [5, 4], rois.lod = [[3, 2]]
        output_channels = 10
        pooled_height = 7
        pooled_width = 7

      Return:
        out.shape = [5, 10, 7, 7], out.lod = [[3, 2]]


参数
::::::::::::

    - **input** (Variable) - 输入特征图，4-D Tensor，格式是 NCHW。其中 N 是 batch_size，C 是输入通道的数量，H 是输入特征图的高度，W 是特征图宽度。数据类型是 float32 或者 float64
    - **rois** (Variable) - 感兴趣区域，2-D LoDTensor，形状为(num_rois, 4)，lod_level 为 1。形式如[x1, y1, x2, y2], ...]，其中(x1, y1)为左上角坐标，(x2, y2)为右下角坐标。数据类型与 input 相同
    - **output_channels** (int) - 输出特征图的通道数。对于共 C 个种类的图像分类任务，output_channels 应该是 ``(C + 1)``，其中 1 代表背景
    - **spatial_scale** (float) - 空间跨度因子，用于将 ``rois`` 中的坐标从其输入尺寸按比例映射到 ``input`` 特征图的尺寸
    - **pooled_height** (int) - 池化输出的高度
    - **pooled_width** (int) - 池化输出的宽度
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 经过 PSROIPooling 之后的结果，形状为(num_rois，output_channels，pooled_height，pooled_width) 的 4 维 LoDTensor，lod_level 为 1，数据类型与 input 相同，与 rois 具有相同的 lod 信息。

返回类型
::::::::::::
  Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.psroi_pool
