.. _cn_api_paddle_vision_ops_roi_align:

roi_align
-------------------------------

.. py:function:: paddle.vision.ops.roi_align(input, boxes, boxes_num, output_size, spatial_scale=1.0, aligned=True, name=None)

感兴趣区域对齐（也称为 RoI Align），是在指定输入的感兴趣区域上执行双线性插值以获得固定大小的特征图（例如7*7），如 Mask R-CNN论文中所述, 请参阅 https://arxiv.org/abs/1703.06870。

参数
:::::::::
    - x (Tensor) - 输入的特征图，形状为(N, C, H, W)。数据类型为float32或float64。
    - boxes (Tensor) - 待执行池化的RoIs(Regions of Interest, 感兴趣区域)的框坐标。它应当是一个形状为(boxes_num, 4)的2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。
    - boxes_num (Tensor) - 该batch中每一张图所包含的框数量。数据类型为int32。
    - output_size (int|Tuple(int, int)) - 池化后输出的尺寸(H, W)，数据类型为int32。如果output_size是单个int类型整数，则H和W都与其相等。
    - spatial_scale (float, 可选) - 空间比例因子，用于将boxes中的坐标从其输入尺寸按比例映射到input特征图的尺寸。
    - aligned (bool）- 默认值为True，表示像素移动框将其坐标移动-0.5，以便与两个相邻像素索引更好地对齐。如果为False，则是使用遗留版本的实现。
    - name (str, 可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

返回
:::::::::
    Tensor，池化后的RoIs，为一个形状是(RoI数量，输出通道数，池化后高度，池化后宽度）的4-D Tensor。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。

代码示例
:::::::::

..  code-block:: python

    import paddle
    from paddle.vision.ops import roi_align

    data = paddle.rand([1, 256, 32, 32])
    boxes = paddle.rand([3, 4])
    boxes[:, 2] += boxes[:, 0] + 3
    boxes[:, 3] += boxes[:, 1] + 4
    boxes_num = paddle.to_tensor([3]).astype('int32')
    align_out = roi_align(data, boxes, boxes_num=boxes_num, output_size=3)
    assert align_out.shape == [3, 256, 3, 3]
