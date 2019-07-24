.. _cn_api_fluid_layers_psroi_pool:

psroi_pool
-------------------------------

.. py:function:: paddle.fluid.layers.psroi_pool(input, rois, output_channels, spatial_scale, pooled_height, pooled_width, name=None)

PSROIPool运算

区分位置的感兴趣区域池化方法（Position sensitive region of interest pooling，也称为PSROIPooling）是对输入的 "感兴趣区域"(RoI)执行按位置的average池化，并将N个按位置评分图（score map）和一个由num_rois个感兴趣区域所组成的列表作为输入。

用于R-FCN的PSROIPooling。 有关更多详细信息，请参阅 https://arxiv.org/abs/1605.06409。

参数：
    - **input** （Variable） - （Tensor），PSROIPoolOp的输入。 输入张量的格式是NCHW。 其中N是批大小batch_size，C是输入通道的数量，H是输入特征图的高度，W是特征图宽度
    - **rois** （Variable） - 要进行池化的RoI（感兴趣区域）。应为一个形状为(num_rois, 4)的二维LoDTensor，其lod level为1。给出[[x1, y1, x2, y2], ...]，(x1, y1)为左上角坐标，(x2, y2)为右下角坐标。
    - **output_channels** （integer） - （int），输出特征图的通道数。 对于共C个种类的对象分类任务，output_channels应该是（C + 1），该情况仅适用于分类任务。
    - **spatial_scale** （float） - （float，default 1.0），乘法空间比例因子，用于将ROI坐标从其输入比例转换为池化使用的比例。默认值：1.0
    - **pooled_height** （integer） - （int，默认值1），池化输出的高度。默认值：1
    - **pooled_width** （integer） - （int，默认值1），池化输出的宽度。默认值：1
    - **name** （str，default None） - 此层的名称。

返回： （Tensor），PSROIPoolOp的输出是形为 (num_rois，output_channels，pooled_h，pooled_w) 的4-D Tensor。

返回类型：  变量（Variable）

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[490, 28, 28], dtype='float32')
    rois = fluid.layers.data(name='rois', shape=[4], lod_level=1, dtype='float32')
    pool_out = fluid.layers.psroi_pool(x, rois, 10, 1.0, 7, 7)





