.. _cn_api_paddle_vision_ops_RoIAlign:

RoIAlign
-------------------------------

.. py:class:: paddle.vision.ops.RoIAlign(output_size, spatial_scale=1.0)

该接口用于构建一个 ``RoIAlign`` 类的可调用对象。请参见 :ref:`cn_api_paddle_vision_ops_roi_align` API。

参数
:::::::::
    - output_size (int|Tuple(int, int)) - 池化后输出的尺寸(H, W)，数据类型为int32。如果output_size是单个int类型整数，则H和W都与其相等。
    - spatial_scale (float，可选) - 空间比例因子，用于将boxes中的坐标从其输入尺寸按比例映射到输入特征图的尺寸，默认值1.0。

形状
:::::::::
    - x: 4-D Tensor，形状为(N, C, H, W)。数据类型为float32或float64。
    - boxes: 2-D Tensor，形状为(boxes_num, 4)。
    - boxes_num: 1-D Tensor。数据类型为int32。
    - output: 4-D tensor，形状为(RoI数量，输出通道数，池化后高度，池化后宽度)。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。

返回
:::::::::
Tensor，形状为(num_boxes, channels, pooled_h, pooled_w)。

代码示例
:::::::::
COPY-FROM: paddle.vision.ops.RoIAlign:code-example1
