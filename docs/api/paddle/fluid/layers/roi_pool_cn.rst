.. _cn_api_fluid_layers_roi_pool:

roi_pool
-------------------------------

.. py:function:: paddle.fluid.layers.roi_pool(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0)





该OP实现了roi池化操作，对非均匀大小的输入执行最大池化，以获得固定大小的特征映射(例如7*7)。

该OP的操作分三个步骤：

    1. 用pooled_width和pooled_height将每个proposal区域划分为大小相等的部分；
    2. 在每个部分中找到最大的值；
    3. 将这些最大值复制到输出缓冲区。

Faster-RCNN使用了roi池化。roi池化的具体原理请参考 https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn

参数
::::::::::::

    - **input** (Variable) - 输入特征，维度为[N,C,H,W]的4D-Tensor，其中N为batch大小，C为输入通道数，H为特征高度，W为特征宽度。数据类型为float32或float64。
    - **rois** (Variable) – 待池化的ROIs (Regions of Interest)，维度为[num_rois,4]的2D-LoDTensor，lod level 为1。给定如[[x1,y1,x2,y2], ...]，其中(x1,y1)为左上点坐标，(x2,y2)为右下点坐标。lod信息记录了每个roi所属的batch_id。
    - **pooled_height** (int，可选) - 数据类型为int32，池化输出的高度。默认值为1。
    - **pooled_width** (int，可选) -  数据类型为int32，池化输出的宽度。默认值为1。
    - **spatial_scale** (float，可选) - 数据类型为float32，用于将ROI coords从输入比例转换为池化时使用的比例。默认值为1.0。

返回
::::::::::::
 池化后的特征，维度为[num_rois, C, pooled_height, pooled_width]的4D-Tensor。

返回类型
::::::::::::
 Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.roi_pool