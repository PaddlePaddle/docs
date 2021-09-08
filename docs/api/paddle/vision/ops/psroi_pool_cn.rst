.. _cn_api_paddle_vision_ops_psroi_pool:

psroi_pool
-------------------------------

.. py:function:: paddle.vision.ops.psroi_pool(input, boxes, boxes_num, output_size, spatial_scale=1.0, name=None)

位置敏感的兴趣区域池化（也称为 PSROIPooling），是在指定输入的感兴趣区域上执行位置敏感的平均池化。它在非均匀大小的输入上执行并获得固定大小的特征图。

PSROIPooling由R-FCN提出。更多详细信息，请参阅 https://arxiv.org/abs/1605.06409。

参数
:::::::::
    - input (Tensor) - 输入的特征图，形状为(N, C, H, W)。
    - boxes (Tensor) - 待执行池化的ROIs(Regions of Interest, 感兴趣区域)的框坐标。它应当是一个形状为(num_rois, 4)的2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。
    - boxes_num (Tensor) - 该batch中每一张图所包含的框数量。
    - output_size (int|Tuple(int, int)) - 池化后输出的尺寸(H, W), 数据类型为int32. 如果output_size是int类型，H和W都与其相等。
    - spatial_scale (float) - 空间比例因子，用于将boxes中的坐标从其输入尺寸按比例映射到input特征图的尺寸。
    - name (str, 可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name` 。


返回
:::::::::
    池化后的ROIs, 为一个形状是(Roi数量，输出通道数，池化后高度，池化后宽度）的4-D Tensor。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。


返回类型
:::::::::
    Tensor


代码示例
:::::::::
    
..  code-block:: python

    import paddle

    x = paddle.uniform([2, 490, 28, 28], dtype='float32')
    boxes = paddle.to_tensor([[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], dtype='float32')
    boxes_num = paddle.to_tensor([1, 2], dtype='int32')
    pool_out = paddle.vision.ops.psroi_pool(x, boxes, boxes_num, 7, 1.0)
