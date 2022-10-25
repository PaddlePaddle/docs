.. _cn_api_paddle_vision_ops_roi_pool:

roi_pool
-------------------------------

.. py:function:: paddle.vision.ops.roi_pool(x, boxes, boxes_num, output_size, spatial_scale=1.0, name=None)

位置敏感的兴趣区域池化（也称为 ROIPooling），是在指定输入的感兴趣区域上执行位置敏感的平均池化，并获得固定大小的特征图。


参数
:::::::::
    - **x** (Tensor) - 输入的特征图，形状为(N, C, H, W)。
    - **boxes** (Tensor) - 待执行池化的 ROIs(Regions of Interest，感兴趣区域)的框坐标。它应当是一个形状为(num_rois, 4)的 2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。
    - **boxes_num** (Tensor) - 该 batch 中每一张图所包含的框数量。
    - **output_size** (int|Tuple[int, int]) - 池化后输出的尺寸(H, W)，数据类型为 int32。如果 output_size 是 int 类型，H 和 W 都与其相等。
    - **spatial_scale** (float，可选) - 空间比例因子，用于将 boxes 中的坐标从其输入尺寸按比例映射到 input 特征图的尺寸，默认值 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
    Tensor，为池化后的 ROIs，为一个形状是(Roi 数量，输出通道数，池化后高度，池化后宽度）的 4-D Tensor。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。


代码示例
:::::::::

COPY-FROM: paddle.vision.ops.roi_pool
