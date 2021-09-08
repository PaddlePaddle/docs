.. _cn_api_paddle_vision_ops_PSRoIPool:

PSRoIPool
-------------------------------

.. py:class:: paddle.vision.ops.PSRoIPool(output_size, spatial_scale=1.0)

该接口用于构建一个 ``PSRoIPool`` 类的可调用对象。请参见 :ref:`cn_api_paddle_vision_ops_psroi_pool` API。

参数
:::::::::
    - output_size (int|Tuple(int, int)) - 池化后输出的尺寸(H, W), 数据类型为int32. 如果output_size是int类型，H和W都与其相等。
    - spatial_scale (float) - 空间比例因子，用于将boxes中的坐标从其输入尺寸按比例映射到input特征图的尺寸。

形状
:::::::::
    - input: 4-D Tensor，形状为(N, C, H, W)。
    - boxes: 2-D Tensor，形状为(num_rois, 4)。
    - boxes_num: 1-D Tensor。
    - output: 4-D tensor，形状为(Roi数量，输出通道数，池化后高度，池化后宽度)。输出通道数等于输入通道数/（池化后高度 * 池化后宽度）。

返回
:::::::::
    无

代码示例
:::::::::
    
..  code-block:: python

    import paddle
    
    psroi_module = paddle.vision.ops.PSRoIPool(7, 1.0)
    x = paddle.uniform([2, 490, 28, 28], dtype='float32')
    boxes = paddle.to_tensor([[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], dtype='float32')
    boxes_num = paddle.to_tensor([1, 2], dtype='int32')
    pool_out = psroi_module(x, boxes, boxes_num)
