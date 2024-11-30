.. _api_guide_pool:

#####
池化
#####

池化的作用是对输入特征做下采样和降低过拟合。降低过拟合是减小输出大小的结果，它同样也减少了后续层中的参数的数量。

池化通常只需要将前一层的特征图作为输入，此外需要一些参数来确定池化具体的操作。在 PaddlePaddle 中我们同样通过设定池化的大小，方式，步长，是否是全局池化，是否使用 cudnn，是否使用 ceil 函数计算输出等参数来选择具体池化的方式。
PaddlePaddle 中有针对定长图像特征的二维(pool2d)、三维卷积(pool3d)，RoI 池化(roi_pool)，以及针对序列的序列池化(sequence_pool)，同时也有池化计算的反向过程，下面先介绍 2D/3D 池化，以及 RoI 池化，再来介绍序列池化。

--------------

1. pool2d/pool3d
------------------------

-  ``input`` : 池化操作接收任何符合 layout 是：\ ``N（batch size）* C(channel size) * H(height) * W(width)``\ 格式的\ ``Tensor``\ 类型作为输入。

-  ``pool_size``\ : 用来确定池化\ ``filter``\ 的大小，即将多大范围内的数据池化为一个值。

-  ``num_channels``\ : 用来确定输入的\ ``channel``\ 数量，如果未设置参数或设置为\ ``None``\ ，其实际值将自动设置为输入的\ ``channel``\ 数量。

-  ``pool_type``\ : 接收\ ``avg``\ 和\ ``max``\ 2 种类型之一作为 pooling 的方式，默认值为\ ``max``\ 。其中\ ``max``\ 意为最大池化，即计算池化\ ``filter``\ 区域内的数据的最大值作为输出；而\ ``avg``\ 意为平均池化，即计算池化\ ``filter``\ 区域内的数据的平均值作为输出。

-  ``pool_stride``\ : 意为池化的\ ``filter``\ 在输入特征图上移动的步长。

-  ``pool_padding``\ : 用来确定池化中\ ``padding``\ 的大小，\ ``padding``\ 的使用是为了对于特征图边缘的特征进行池化，选择不同的\ ``pool_padding``\ 大小确定了在特征图边缘增加多大区域的补零。从而决定边缘特征被池化的程度。

-  ``global_pooling``\ : 意为是否使用全局池化，全局池化是指使用和特征图大小相同的\ ``filter``\ 来进行池化，同样这个过程也可以使用平均池化或者最大池化来做为池化的方式，全局池化通常会用来替换全连接层以大量减少参数防止过拟合。

-  ``use_cudnn``\ : 选项可以来选择是否使用 cudnn 来优化计算池化速度。

-  ``ceil_mode``\ : 是否使用 ceil 函数计算输出高度和宽度。\ ``ceil mode``\ 意为天花板模式，是指会把特征图中不足\ ``filter size``\ 的边给保留下来，单独另算，或者也可以理解为在原来的数据上补充了值为-NAN 的边。而 floor 模式则是直接把不足\ ``filter size``\ 的边给舍弃了。具体计算公式如下：

    -  非\ ``ceil_mode``\ 下:\ ``输出大小 = (输入大小 - filter size + 2 * padding) / stride（步长） + 1``

    -  ``ceil_mode``\ 下:\ ``输出大小 = (输入大小 - filter size + 2 * padding + stride - 1) / stride + 1``



api 汇总：

- :ref:`cn_api_fluid_layers_pool2d`
- :ref:`cn_api_fluid_layers_pool3d`


2. roi_pool
------------------

``roi_pool``\ 一般用于检测网络中，将输入特征图依据候选框池化到特定的大小。

-  ``rois``\ : 接收\ ``DenseTensor``\ 类型来表示需要池化的 Regions of Interest，关于 RoI 的解释请参考\ `论文 <https://arxiv.org/abs/1506.01497>`__

-  ``pooled_height`` 和 ``pooled_width``\ : 这里可以接受非正方的池化窗口大小

-  ``spatial_scale``\ : 用作设定缩放 RoI 和原图缩放的比例，注意，这里的设定需要用户自行计算 RoI 和原图的实际缩放比例。


api 汇总：

- :ref:`cn_api_fluid_layers_roi_pool`


3. sequence_pool
--------------------

``sequence_pool``\ 是一个用作对于不等长序列进行池化的接口，它将每一个实例的全部时间步的特征进行池化，它同样支持
``average``, ``sum``, ``sqrt`` 和\ ``max``\ 4 种类型之一作为 pooling 的方式。 其中:

-  ``average``\ 是对于每一个时间步内的数据求和后分别取平均值做为池化的结果。

-  ``sum``\ 则是对每一个时间步内的数据分别求和作为池化的结果。

-  ``sqrt``\ 则是对每一个时间步内的数据分别求和再分别取平方根作为池化的结果。

-  ``max``\ 则是对每一个时间步内的数据分别求取最大值作为池化的结果。

api 汇总：

- :ref:`cn_api_fluid_layers_sequence_pool`
