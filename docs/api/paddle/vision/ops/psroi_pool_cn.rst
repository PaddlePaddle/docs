.. _cn_api_paddle_vision_ops_psroi_pool:

psroi_pool
-------------------------------

.. py:function:: paddle.vision.ops.psroi_pool(x, boxes, boxes_num, output_size, spatial_scale=1.0, name=None)

位置敏感的兴趣区域池化（也称为 PSROIPooling），是在指定输入的感兴趣区域上执行位置敏感的平均池化。它在非均匀大小的输入上执行并获得固定大小的特征图。

PSROIPooling 由 R-FCN 提出。更多详细信息，请参阅 https://arxiv.org/abs/1605.06409。

参数
:::::::::
    - **x** (Tensor) - 输入的特征图，形状为(N, C, H, W)，数据类型为 float32 或 float64。
    - **boxes** (Tensor) - 待执行池化的 ROIs(Regions of Interest，感兴趣区域)的框坐标。它应当是一个形状为(num_rois, 4)的 2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。
    - **boxes_num** (Tensor) - 该 batch 中每一张图所包含的框数量。
    - **output_size** (int|Tuple(int, int)) - 池化后输出的尺寸(H, W)，数据类型为 int32。如果 output_size 是 int 类型，H 和 W 都与其相等。
    - **spatial_scale** (float，可选) - 空间比例因子，用于将 boxes 中的坐标从其输入尺寸按比例映射到输入特征图的尺寸。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
    4-D Tensor。池化后的 ROIs，其形状是(Roi 数量，输出通道数，池化后高度，池化后宽度）。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。

代码示例
:::::::::

COPY-FROM: paddle.vision.ops.psroi_pool
