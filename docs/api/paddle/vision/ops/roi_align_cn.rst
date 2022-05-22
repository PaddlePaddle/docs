.. _cn_api_paddle_vision_ops_roi_align:

roi_align
-------------------------------

.. py:function:: paddle.vision.ops.roi_align(x, boxes, boxes_num, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=True, name=None)

RoI Align是在指定输入的感兴趣区域上执行双线性插值以获得固定大小的特征图（例如7*7），如 Mask R-CNN论文中所述。

论文参考：`Mask R-CNN <https://arxiv.org/abs/1703.06870>`_ 。

参数
:::::::::
    - **x (Tensor)** - 输入的特征图，形状为(N, C, H, W)。数据类型为float32或float64。
    - **boxes (Tensor)** - 待执行池化的RoIs(Regions of Interest)的框坐标。它应当是一个形状为(boxes_num, 4)的2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。
    - **boxes_num (Tensor)** - 该batch中每一张图所包含的框数量。数据类型为int32。
    - **output_size (int|Tuple(int, int))** - 池化后输出的尺寸(H, W)，数据类型为int32。如果output_size是单个int类型整数，则H和W都与其相等。
    - **spatial_scale (float32)** - 空间比例因子，用于将boxes中的坐标从其输入尺寸按比例映射到input特征图的尺寸。
    - **sampling_ratio (int32)** – 插值网格中用于计算每个池化输出条柱的输出值的采样点数。如果大于0，则使用每个条柱的精确采样点。如果小于或等于0，则使用自适应数量的网格点（计算为 ``ceil(roi_width / output_width)``，高度同理）。默认值：-1。
    - **aligned (bool，可选）**- 默认值为True，表示像素移动框将其坐标移动-0.5，以便与两个相邻像素索引更好地对齐。如果为False，则是使用遗留版本的实现。
    - **name (str，可选）**- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

返回
:::::::::
    Tensor，池化后的RoIs，为一个形状是(RoI数量，输出通道数，池化后高度，池化后宽度）的4-D Tensor。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。

代码示例
:::::::::
COPY-FROM: paddle.vision.ops.roi_align:code-example1
