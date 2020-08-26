.. _cn_api_fluid_layers_prroi_pool:

prroi_pool
-------------------------------

.. py:function:: paddle.fluid.layers.prroi_pool(input, rois, output_channels, spatial_scale, pooled_height, pooled_width, name=None)

:alias_main: paddle.nn.functional.prroi_pool
:alias: paddle.nn.functional.prroi_pool,paddle.nn.functional.vision.prroi_pool
:old_api: paddle.fluid.layers.prroi_pool



PRROIPool运算

精确区域池化方法（Precise region of interest pooling，也称为PRROIPooling）是对输入的 "感兴趣区域"(RoI)执行插值处理，将离散的特征图数据映射到一个连续空间，使用二重积分再求均值的方式实现Pooling。

通过积分方式计算ROI特征，反向传播时基于连续输入值计算梯度，使得反向传播连续可导的PRROIPooling。 有关更多详细信息，请参阅 https://arxiv.org/abs/1807.11590。

参数：
    - **input** （Variable） - （Tensor），PRROIPoolOp的输入。 输入张量的格式是NCHW。 其中N是批大小batch_size，C是输入通道的数量，H是输入特征图的高度，W是特征图宽度
    - **rois** （Variable） - 要进行池化的RoI（感兴趣区域）。应为一个形状为(num_rois, 4)的二维LoDTensor，其lod level为1。给出[[x1, y1, x2, y2], ...]，(x1, y1)为左上角坐标，(x2, y2)为右下角坐标。
    - **output_channels** （integer） - （int），输出特征图的通道数。 对于共C个种类的对象分类任务，output_channels应该是（C + 1），该情况仅适用于分类任务。
    - **spatial_scale** （float） - （float，default 1.0），乘法空间比例因子，用于将ROI坐标从其输入比例转换为池化使用的比例。默认值：1.0
    - **pooled_height** （integer） - （int，默认值1），池化输出的高度。默认值：1
    - **pooled_width** （integer） - （int，默认值1），池化输出的宽度。默认值：1
    - **name** （str，default None） - 此层的名称。

返回： （Tensor），PRROIPoolOp的输出是形为 (num_rois，output_channels，pooled_h，pooled_w) 的4-D Tensor。

返回类型：  变量（Variable）

**代码示例：**

.. code-block:: python

    ## prroi_pool without batch_roi_num
    import paddle.fluid as fluid
    x = fluid.data(name='x', shape=[None, 490, 28, 28], dtype='float32')
    rois = fluid.data(name='rois', shape=[None, 4], lod_level=1, dtype='float32')
    pool_out = fluid.layers.prroi_pool(x, rois, 1.0, 7, 7)

    ## prroi_pool with batch_roi_num
    batchsize=4
    x2 = fluid.data(name='x2', shape=[batchsize, 490, 28, 28], dtype='float32')
    rois2 = fluid.data(name='rois2', shape=[batchsize, 4], dtype='float32')
    batch_rois_num = fluid.data(name='rois_nums', shape=[batchsize], dtype='int64')
    pool_out2 = fluid.layers.prroi_pool(x2, rois2, 1.0, 7, 7, batch_roi_nums=batch_rois_num)





