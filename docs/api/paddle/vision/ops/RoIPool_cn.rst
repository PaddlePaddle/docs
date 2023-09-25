.. _cn_api_paddle_vision_ops_RoIPool:

RoIPool
-------------------------------

.. py:class:: paddle.vision.ops.RoIPool(output_size, spatial_scale=1.0)

构建一个 ``RoIPool`` 类的可调用对象。请参见 :ref:`cn_api_paddle_vision_ops_roi_pool` API。

参数
:::::::::
    - **output_size** (int|Tuple[int, int]) - 池化后输出的尺寸(H, W)，数据类型为 int32。如果 output_size 是 int 类型，H 和 W 都与其相等。
    - **spatial_scale** (float，可选) - 空间比例因子，用于将 boxes 中的坐标从其输入尺寸按比例映射到 input 特征图的尺寸，默认值 1.0。


返回
:::::::::
    Tensor，为池化后的 ROIs，为一个形状是(Roi 数量，输出通道数，池化后高度，池化后宽度）的 4-D Tensor。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。

代码示例
:::::::::

COPY-FROM: paddle.vision.ops.RoIPool
