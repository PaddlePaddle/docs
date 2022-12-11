.. _cn_api_paddle_vision_ops_RoIAlign:

RoIAlign
-------------------------------

.. py:class:: paddle.vision.ops.RoIAlign(output_size, spatial_scale=1.0)

构建一个 ``RoIAlign`` 类的可调用对象。请参见 :ref:`cn_api_paddle_vision_ops_roi_align` API。

参数
:::::::::
    - **output_size** (int|Tuple[int, int]) - 池化后输出的尺寸(H, W)，数据类型为 int32。如果 output_size 是单个 int 类型整数，则 H 和 W 都与其相等。
    - **spatial_scale** (float，可选) - 空间比例因子，用于将 boxes 中的坐标从其输入尺寸按比例映射到输入特征图的尺寸，默认值 1.0。


返回
:::::::::
    形状是(RoI 数量，输出通道数，池化后高度，池化后宽度）的 4-D Tensor 。

代码示例
:::::::::

COPY-FROM: paddle.vision.ops.RoIAlign
