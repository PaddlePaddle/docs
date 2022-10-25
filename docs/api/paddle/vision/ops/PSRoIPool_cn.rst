.. _cn_api_paddle_vision_ops_PSRoIPool:

PSRoIPool
-------------------------------

.. py:class:: paddle.vision.ops.PSRoIPool(output_size, spatial_scale=1.0)

构建一个 ``PSRoIPool`` 类的可调用对象。请参见 :ref:`cn_api_paddle_vision_ops_psroi_pool` API。

参数
:::::::::
    - **output_size** (int|Tuple(int, int)) - 池化后输出的尺寸(H, W), 数据类型为 int32. 如果 output_size 是 int 类型，H 和 W 都与其相等。
    - **spatial_scale** (float，可选) - 空间比例因子，用于将 boxes 中的坐标从其输入尺寸按比例映射到输入特征图的尺寸。

形状
:::::::::
    - x: 4-D Tensor，形状为(N, C, H, W)。数据类型为 float32 或 float64。
    - boxes: 2-D Tensor，形状为(num_rois, 4)。
    - boxes_num: 1-D Tensor。
    - output: 4-D tensor，形状为(Roi 数量，输出通道数，池化后高度，池化后宽度)。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。

返回
:::::::::
无。

代码示例
:::::::::

COPY-FROM: paddle.vision.ops.PSRoIPool
