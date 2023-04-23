.. _cn_api_fluid_layers_roi_align:

roi_align
-------------------------------

.. py:function:: paddle.fluid.layers.roi_align(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0, sampling_ratio=-1, name=None)




**实现RoIAlign操作。**

Region of Interests align(直译：有意义、有价值选区对齐) 用于实现双线性插值，它可以将不均匀大小的输入
变为固定大小的特征图(feature map)。

该运算通过 ``pooled_width`` 和 ``pooled_height`` 将每个推荐区域划分为等大小分块。位置保持不变。

在每个RoI分块中，分别取sampling_ratio个点（若为-1则取框内所有点），每个点通过双线性插值直接计算得到坐标。再对分块内取的点取平均值作为小框的坐标值。坐标对齐有误的问题。

参数
::::::::::::

  - **input** (Variable) – 维度为[N,C,H,W]的4-D Tensor，N 为batch大小，C 为输入通道的个数，H 特征高度，W 特征宽度。数据类型为float32或float64。
  - **rois** (Variable) – 维度为[num_rois,4]2-D LoDTensor，数据类型为float32或float64。待池化的ROIs (Regions of Interest)，lod level 为1。给定比如[[x1,y1,x2,y2], ...],(x1,y1)为左上点坐标，(x2,y2)为右下点坐标。
  - **pooled_height** (int32，可选) – 池化后的输出高度，默认值为1。
  - **pooled_width** (int32，可选) – 池化后的输出宽度，默认值为1。
  - **spatial_scale** (float32，可选) – 乘法性质空间标尺因子，池化时，将RoI坐标变换至运算采用的标度，默认值为1.0。
  - **sampling_ratio** (int32) – 插值格中采样点的数目。如果它 <=0，它们将自适应 ``roi_width`` 和 ``pooled_w``，在高度上也是同样的道理。默认值为-1
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。 


返回
::::::::::::
表示RoI align输出的LoDTensor，数据类型为float32或float64，维度为 (num_rois, channels, pooled_h, pooled_w)


返回类型
::::::::::::
Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.roi_align