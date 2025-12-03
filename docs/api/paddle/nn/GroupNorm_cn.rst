.. _cn_api_paddle_nn_GroupNorm:

GroupNorm
-------------------------------

.. py:class:: paddle.nn.GroupNorm(num_groups, num_channels, epsilon=1e-05, *, affine=True, device=None, dtype=None, weight_attr=None, bias_attr=None, data_format='NCHW', name=None)

**Group Normalization 层**

构建 ``GroupNorm`` 类的一个可调用对象，具体用法参照 ``代码示例``。其中实现了组归一化层的功能。更多详情请参考：`Group Normalization <https://arxiv.org/abs/1803.08494>`_ 。

参数
::::::::::::

    - **num_groups** (int) - 从通道中分离出来的 ``group`` 的数目。
    - **num_channels** (int) - 输入的通道数。
    - **epsilon** (float，可选) - 为防止方差除零，增加一个很小的值。默认值：1e-05。
      ``别名: eps``
    - **affine** (bool，可选) - 该模块是否具有可学习的仿射参数（weight 和 bias）。如果设置为 False，将不会创建可学习的参数，无论 weight_attr 和 bias_attr 如何设置。默认值：True。
      ``注意： 此参数必须以关键字参数的形式传入``
    - **device** (PlaceLike，可选) - 计算发生的设备。默认值：None。
      ``注意： 此参数必须以关键字参数的形式传入``
    - **dtype** (DTypeLike，可选) - 权重和偏置的数据类型。默认值：None。
      ``注意： 此参数必须以关键字参数的形式传入``
    - **weight_attr** (ParamAttr|bool，可选) - 指定权重参数属性的对象。如果为 False，表示参数不学习。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
      ``注意： 此参数必须以关键字参数的形式传入``
    - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数属性的对象。如果为 False，表示参数不学习。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
      ``注意： 此参数必须以关键字参数的形式传入``
    - **data_format** (string，可选) - 支持 “NCL”，“NCHW”，“NCDHW”，“NLC”，“NHWC”，“NDHWC” 格式。默认值：“NCHW”。
      ``注意： 此参数必须以关键字参数的形式传入``
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
      ``注意： 此参数必须以关键字参数的形式传入``

形状
::::::::::::

    - input：形状为 (批大小，通道数，\*) 或 (批大小，\*，通道数) 的 Tensor。
    - output：和输入形状一样。

代码示例
::::::::::::

COPY-FROM: paddle.nn.GroupNorm
