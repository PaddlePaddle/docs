.. _cn_api_fluid_layers_prroi_pool:

prroi_pool
-------------------------------

.. py:function:: paddle.fluid.layers.prroi_pool(input, rois, output_channels, spatial_scale, pooled_height, pooled_width, name=None)




PRROIPool 运算

精确区域池化方法（Precise region of interest pooling，也称为 PRROIPooling）是对输入的 "感兴趣区域"(RoI)执行插值处理，将离散的特征图数据映射到一个连续空间，使用二重积分再求均值的方式实现 Pooling。

通过积分方式计算 ROI 特征，反向传播时基于连续输入值计算梯度，使得反向传播连续可导的 PRROIPooling。有关更多详细信息，请参阅 https://arxiv.org/abs/1807.11590。

参数
::::::::::::

    - **input** （Variable） - （Tensor），PRROIPoolOp 的输入。输入 Tensor 的格式是 NCHW。其中 N 是批大小 batch_size，C 是输入通道的数量，H 是输入特征图的高度，W 是特征图宽度
    - **rois** （Variable） - 要进行池化的 RoI（感兴趣区域）。应为一个形状为(num_rois, 4)的二维 Tensor，其 lod level 为 1。给出[[x1, y1, x2, y2], ...]，(x1, y1)为左上角坐标，(x2, y2)为右下角坐标。
    - **output_channels** （integer） - （int），输出特征图的通道数。对于共 C 个种类的对象分类任务，output_channels 应该是（C + 1），该情况仅适用于分类任务。
    - **spatial_scale** （float） - （float，default 1.0），乘法空间比例因子，用于将 ROI 坐标从其输入比例转换为池化使用的比例。默认值：1.0
    - **pooled_height** （integer） - （int，默认值 1），池化输出的高度。默认值：1
    - **pooled_width** （integer） - （int，默认值 1），池化输出的宽度。默认值：1
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 （Tensor），PRROIPoolOp 的输出是形为 (num_rois，output_channels，pooled_h，pooled_w) 的 4-D Tensor。

返回类型
::::::::::::
  变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.prroi_pool
