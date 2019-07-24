.. _cn_api_fluid_layers_roi_perspective_transform:

roi_perspective_transform
-------------------------------

.. py:function:: paddle.fluid.layers.roi_perspective_transform(input, rois, transformed_height, transformed_width, spatial_scale=1.0)

**ROI perspective transform操作符**

参数：
    - **input** (Variable) - ROI Perspective TransformOp的输入。输入张量的形式为NCHW。N是批尺寸，C是输入通道数，H是特征高度，W是特征宽度
    - **rois** (Variable) - 用来处理的ROIs，应该是shape的二维LoDTensor(num_rois,8)。给定[[x1,y1,x2,y2,x3,y3,x4,y4],...],(x1,y1)是左上角坐标，(x2,y2)是右上角坐标，(x3,y3)是右下角坐标，(x4,y4)是左下角坐标
    - **transformed_height** (integer) - 输出的高度
    - **transformed_width** (integer) – 输出的宽度
    - **spatial_scale** (float) - 空间尺度因子，用于缩放ROI坐标，默认：1.0。

返回：
 ``ROIPerspectiveTransformOp`` 的输出，它是一个4维张量，形为 (num_rois,channels,transformed_h,transformed_w)

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name='x', shape=[256, 28, 28], dtype='float32')
    rois = fluid.layers.data(name='rois', shape=[8], lod_level=1, dtype='float32')
    out = fluid.layers.roi_perspective_transform(x, rois, 7, 7, 1.0)







