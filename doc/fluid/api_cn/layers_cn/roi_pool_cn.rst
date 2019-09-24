.. _cn_api_fluid_layers_roi_pool:

roi_pool
-------------------------------

.. py:function:: paddle.fluid.layers.roi_pool(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0)


roi池化是对非均匀大小的输入执行最大池化，以获得固定大小的特征映射(例如7*7)。

该operator有三个步骤:

    1. 用pooled_width和pooled_height将每个区域划分为大小相等的部分
    2. 在每个部分中找到最大的值
    3. 将这些最大值复制到输出缓冲区

Faster-RCNN.使用了roi池化。roi关于roi池化请参考 https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn

参数:
    - **input** (Variable) - 张量，ROIPoolOp的输入。输入张量的格式是NCHW。其中N为batch大小，C为输入通道数，H为特征高度，W为特征宽度
    - **rois** (Variable) – 待池化的ROIs (Regions of Interest)，形为（num_rois,4）的2D张量，lod level 为1。给定比如[[x1,y1,x2,y2], ...],(x1,y1)为左上点坐标，(x2,y2)为右下点坐标。
    - **pooled_height** (integer) - (int，默认1)，池化输出的高度。默认:1
    - **pooled_width** (integer) -  (int，默认1) 池化输出的宽度。默认:1
    - **spatial_scale** (float) - (float，默认1.0)，用于将ROI coords从输入比例转换为池化时使用的比例。默认1.0

返回: (张量)，ROIPoolOp的输出是一个shape为(num_rois, channel, pooled_h, pooled_w)的4d张量。

返回类型: 变量（Variable）


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
     
  x = fluid.layers.data(
            name='x', shape=[8, 112, 112], dtype='float32')
  rois = fluid.layers.data(
            name='roi', shape=[4], lod_level=1, dtype='float32')
  pool_out = fluid.layers.roi_pool(
            input=x,
            rois=rois,
            pooled_height=7,
            pooled_width=7,
            spatial_scale=1.0)









