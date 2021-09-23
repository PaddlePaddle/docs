.. _cn_api_paddle_vision_ops_RoIPool:

RoIPool
-------------------------------

.. py:class:: paddle.vision.ops.RoIPool(output_size, spatial_scale=1.0)

该接口用于构建一个 ``RoIPool`` 类的可调用对象。请参见 :ref:`cn_api_paddle_vision_ops_roi_pool` API。

参数
:::::::::
    - output_size (int|Tuple(int, int)) - 池化后输出的尺寸(H, W), 数据类型为int32. 如果output_size是int类型，H和W都与其相等。
    - spatial_scale (float) - 空间比例因子，用于将boxes中的坐标从其输入尺寸按比例映射到input特征图的尺寸。

形状
:::::::::
    - x: 4-D Tensor，形状为(N, C, H, W)。
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
    from paddle.vision.ops import RoIPool
    
    data = paddle.rand([1, 256, 32, 32])
    boxes = paddle.rand([3, 4])
    boxes[:, 2] += boxes[:, 0] + 3
    boxes[:, 3] += boxes[:, 1] + 4
    boxes_num = paddle.to_tensor([3]).astype('int32')
    roi_pool = RoIPool(output_size=(4, 3))
    pool_out = roi_pool(data, boxes, boxes_num)
    assert pool_out.shape == [3, 256, 4, 3], ''