.. _cn_api_paddle_vision_ops_roi_align:

roi_align
-------------------------------

.. py:function:: paddle.vision.ops.roi_align(x, boxes, boxes_num, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=True, name=None)

实现 roi_align 层，感兴趣区域对齐算子（也称为 RoI 对齐）是对非均匀大小的输入执行双线性插值，以获得固定大小的特征图（例如 7*7），如 `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_ 中所述。将每个区域分成大小相等的部分，其中包含 pooled_width 和 pooled_height。保留原始位置。在每个 ROI 仓中，通过双线性插值直接计算四个规则采样位置的值。输出四个位置的平均值。因此避免了错位问题。


参数
:::::::::
    - **x** (Tensor) - 输入的特征图，形状为(N, C, H, W)。N 是批数据大小，C 是特征图个数，H 是特征图高度，W 是特征图宽度。数据类型为 float32 或 float64。
    - **boxes** (Tensor) - 待执行池化的 RoIs(Regions of Interest)的框坐标。它应当是一个形状为(boxes_num, 4)的 2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。
    - **boxes_num** (Tensor) - 该 batch 中每一张图所包含的框数量。数据类型为 int32。
    - **output_size** (int|Tuple(int, int)) - 池化后输出的尺寸(H, W)，数据类型为 int32。如果 output_size 是单个 int 类型整数，则 H 和 W 都与其相等。
    - **spatial_scale** (float32，可选) - 空间比例因子，用于将 boxes 中的坐标从其输入尺寸按比例映射到 input 特征图的尺寸。
    - **sampling_ratio** (int32，可选) - 插值网格中用于计算每个池化输出条柱的输出值的采样点数。如果大于 0，则使用每个条柱的精确采样点。如果小于或等于 0，则使用自适应数量的网格点（计算为 ``ceil(roi_width / output_width)``，高度同理）。默认值：-1。
    - **aligned** (bool，可选) - 默认值为 True，表示像素移动框将其坐标移动-0.5，以便与两个相邻像素索引更好地对齐。如果为 False，则是使用遗留版本的实现。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    Tensor，池化后的 RoIs，为一个形状是(RoI 数量，输出通道数，池化后高度，池化后宽度）的 4-D Tensor。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。

代码示例
:::::::::

COPY-FROM: paddle.vision.ops.roi_align
