.. _cn_api_fluid_layers_roi_align:

roi_align
-------------------------------

.. py:function:: paddle.fluid.layers.roi_align(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0, sampling_ratio=-1, name=None)

**实现RoIAlign操作。**

Region of Interests align(直译：有意义、有价值选区对齐) 用于实现双线性插值，它可以将不均匀大小的输入
变为固定大小的特征图(feature map)。

该运算通过 ``pooled_width`` 和 ``pooled_height`` 将每个推荐区域划分为等大小分块。位置保持不变。

在每个RoI框中，四个常取样位置会通过双线性插值直接计算。输出为这四个位置的平均值从而解决不对齐问题。

参数:
  - **input** (Variable) – (Tensor) 该运算的的输入张量，形为(N,C,H,W)。其中 N 为batch大小, C 为输入通道的个数, H 特征高度, W 特征宽度
  - **rois** (Variable) – 待池化的ROIs (Regions of Interest)
  - **pooled_height** (integer) – (默认为1), 池化后的输出高度
  - **pooled_width** (integer) – (默认为1), 池化后的输出宽度
  - **spatial_scale** (float) – (默认为1.0),乘法性质空间标尺因子，池化时，将RoI坐标变换至运算采用的标度
  - **sampling_ratio** (intger) – (默认为-1),插值格中采样点的数目。 如果它 <=0, 它们将自适应 ``roi_width`` 和 ``pooled_w`` , 在高度上也是同样的道理。

返回：一个形为 (num_rois, channels, pooled_h, pooled_w) 的四维张量

返回类型：Variable

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











