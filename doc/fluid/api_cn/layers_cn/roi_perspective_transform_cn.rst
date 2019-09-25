.. _cn_api_fluid_layers_roi_perspective_transform:

roi_perspective_transform
-------------------------------

.. py:function:: paddle.fluid.layers.roi_perspective_transform(input, rois, transformed_height, transformed_width, spatial_scale=1.0)

该OP对RoI区域做透视变换，将不规则的RoI区域变成固定大小的矩形区域，透视变换是线性代数里面的一种基础变换。

参数：
    - **input** (Variable) - 输入特征图，4-D Tensor，格式为NCHW。N是batch_size，C是输入通道数，H是特征图高度，W是特征图宽度。数据类型是float32
    - **rois** (Variable) - 感兴趣区域，2D-LoDTensor，形状是(num_rois,8)，lod_level为1。其数据形式是[[x1,y1,x2,y2,x3,y3,x4,y4], ...]，其中(x1,y1)是左上角坐标，(x2,y2)是右上角坐标，(x3,y3)是右下角坐标，(x4,y4)是左下角坐标。数据类型与 ``input`` 相同
    - **transformed_height** (int) - 输出的高度
    - **transformed_width** (int) – 输出的宽度
    - **spatial_scale** (float，可选) - 空间尺度因子，用于缩放ROI坐标，浮点数。默认值1.0

返回： 由三个变量构成的元组 (out, mask, transform_matrix)
 - ``out`` : ``ROIPerspectiveTransformOp`` 的输出，4D-LoDTensor，形状是(num_rois,channels,transformed_height,transformed_width)，lod_level为1
 - ``mask`` : ``ROIPerspectiveTransformOp`` 的掩码，4D-LoDTensor，形状是(num_rois,1,transformed_height,transformed_width)，lod_level为1
 - ``transform_matrix`` : ``ROIPerspectiveTransformOp`` 的转换矩阵，2D-LoDTensor，形状是(num_rois,9)，lod_level为1

返回类型：  元组

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name='x', shape=[256, 28, 28], dtype='float32')
    rois = fluid.layers.data(name='rois', shape=[8], lod_level=1, dtype='float32')
    out, mask, transform_matrix = fluid.layers.roi_perspective_transform(x, rois, 7, 7, 1.0)







