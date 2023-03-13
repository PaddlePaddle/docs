.. _cn_api_fluid_layers_density_prior_box:

density_prior_box
-------------------------------

.. py:function:: paddle.fluid.layers.density_prior_box(input, image, densities=None, fixed_sizes=None, fixed_ratios=None, variance=[0.1, 0.1, 0.2, 0.2], clip=False, steps=[0.0, 0.0], offset=0.5, flatten_to_2d=False, name=None)





该 OP 为 SSD 算法(Single Shot MultiBox Detector)生成 density prior box，在每个 ``input`` 的位置产生 N 个候选框，其中，N 由 ``densities`` , ``fixed_sizes`` 和 ``fixed_ratios`` 来计算。生成的每个输入位置附近的候选框中心（网格点）由 ``densities`` 和 ``density prior box`` 的数量计算，其中 ``density prior box`` 的数量由 ``fixed_sizes`` 和 ``fixed_ratios`` 决定。``fixed_sizes`` 和 ``densities`` 的大小一致。

.. math::

  N\_density\_prior\_box =sum(N\_fixed\_ratios * {densities\_i}^2)


参数
::::::::::::

  - **input** (Variable) - 形状为 NCHW 的 4-D Tensor，数据类型为 float32 或 float64。
  - **image** (Variable) - 输入图像，形状为 NCHW 的 4-D Tensor，数据类型为 float32 或 float64。
  - **densities** (list|tuple|None) - 生成的 density prior boxes 的 densities，此属性应该是一个整数列表或数组。默认值为 None。
  - **fixed_sizes** (list|tuple|None) - 生成的 density prior boxes 的大小，此属性应该为和 :attr:`densities` 有同样长度的列表或数组。默认值为 None。
  - **fixed_ratios** (list|tuple|None) - 生成的 density prior boxes 的比值，如果该属性未被设置，同时 :attr:`densities` 和 :attr:`fix_sizes` 被设置，则 :attr:`aspect_ratios` 被用于生成 density prior boxes
  - **variance** (list|tuple) - 将被用于 density prior boxes 编码的方差，默认值为：[0.1, 0.1, 0.2, 0.2]
  - **clip** (bool) - 是否裁剪超出范围的 box。默认值：False
  - **step** (list|tuple) - Prior boxes 在宽度和高度的步长，如果 step[0]等于 0.0 或 step[1]等于 0.0, input 的 the density prior boxes 的高度/宽度的步长将被自动计算。默认值：Default: [0., 0.]
  - **offset** (float) - Prior boxes 中心偏移值，默认为：0.5
  - **flatten_to_2d** (bool) - 是否将 output prior boxes 和方差 ``flatten`` 至 2-D，其中第二个 dim 为 4。默认值：False
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
含有两个变量的元组，包括：
  候选框：

    当 flatten_to_2d 为 False 时，形状为[H, W, num_priors, 4]的 4-D Tensor。
    当 flatten_to_2d 为 True 时，形式为[H * W * num_priors, 4]的 2-D Tensor。
    其中，H 是输入的高度，W 是输入的宽度，num_priors 是输入中每个位置的候选框数。

  候选框的方差：

    当 flatten_to_2d 为 False 时，形状为[H, W, num_priors, 4]的 4-D Tensor。
    当 flatten_to_2d 为 True 时，形式为[H * W * num_priors, 4]的 2-D Tensor。
    其中，H 是输入的高度，W 是输入的宽度，num_priors 是输入中每个位置的候选框数。

返回类型
::::::::::::
元组

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.density_prior_box
