.. _cn_api_fluid_layers_roi_align:

roi_align
-------------------------------

.. py:function:: paddle.fluid.layers.roi_align(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0, sampling_ratio=-1, name=None)

**实现RoIAlign操作。**

Region of Interests align(直译：有意义、有价值选区对齐) 用于实现双线性插值，它可以将不均匀大小的输入
变为固定大小的特征图(feature map)。

该运算通过 ``pooled_width`` 和 ``pooled_height`` 将每个推荐区域划分为等大小分块。位置保持不变。

在每个RoI分块中，分别取sampling_ratio个点（若为-1则取框内所有点），每个点通过双线性插值直接计算得到坐标。再对分块内取的点取平均值作为小框的坐标值。坐标对齐有误的问题。

参数:
  - **input** (Tensor) – 数据类型为float32, float64的Tensor。形为(N,C,H,W)。其中 N 为batch大小, C 为输入通道的个数, H 特征高度, W 特征宽度
  - **rois** (LoDTensor) – 数据类型为float32, float64的LoDTensor。待池化的ROIs (Regions of Interest)，形为（num_rois,4）的2D张量，lod level 为1。给定比如[[x1,y1,x2,y2], ...],(x1,y1)为左上点坐标，(x2,y2)为右下点坐标。
  - **pooled_height** (integer) – (默认为1), 池化后的输出高度
  - **pooled_width** (integer) – (默认为1), 池化后的输出宽度
  - **spatial_scale** (float) – (默认为1.0),乘法性质空间标尺因子，池化时，将RoI坐标变换至运算采用的标度
  - **sampling_ratio** (intger) – (默认为-1),插值格中采样点的数目。 如果它 <=0, 它们将自适应 ``roi_width`` 和 ``pooled_w`` , 在高度上也是同样的道理。
  - **name** （str|None）- 这一层的名称（可选）。默认为None。

返回：Variable（LoDTensor），数据类型为float32, float64的LoDTensor。形状为 (num_rois, channels, pooled_h, pooled_w)


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(
            name='data', shape=[256, 32, 32], dtype='float32')
    rois = fluid.layers.data(
            name='rois', shape=[4], dtype='float32')
    align_out = fluid.layers.roi_align(input=x,
                                       rois=rois,
                                       pooled_height=7,
                                       pooled_width=7,
                                       spatial_scale=0.5,
                                       sampling_ratio=-1)

