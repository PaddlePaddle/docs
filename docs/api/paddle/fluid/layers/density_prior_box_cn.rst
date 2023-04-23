.. _cn_api_fluid_layers_density_prior_box:

density_prior_box
-------------------------------

.. py:function:: paddle.fluid.layers.density_prior_box(input, image, densities=None, fixed_sizes=None, fixed_ratios=None, variance=[0.1, 0.1, 0.2, 0.2], clip=False, steps=[0.0, 0.0], offset=0.5, flatten_to_2d=False, name=None)





该OP为SSD算法(Single Shot MultiBox Detector)生成density prior box，在每个 ``input`` 的位置产生N个候选框，其中，N由 ``densities`` , ``fixed_sizes`` 和 ``fixed_ratios`` 来计算。生成的每个输入位置附近的候选框中心（网格点）由 ``densities`` 和 ``density prior box`` 的数量计算，其中 ``density prior box`` 的数量由 ``fixed_sizes`` 和 ``fixed_ratios`` 决定。``fixed_sizes`` 和 ``densities`` 的大小一致。

.. math::

  N\_density\_prior\_box =sum(N\_fixed\_ratios * {densities\_i}^2)


参数
::::::::::::

  - **input** (Variable) - 形状为NCHW的4-D Tensor，数据类型为float32或float64。
  - **image** (Variable) - 输入图像，形状为NCHW的4-D Tensor，数据类型为float32或float64。
  - **densities** (list|tuple|None) - 生成的density prior boxes的densities，此属性应该是一个整数列表或数组。默认值为None。
  - **fixed_sizes** (list|tuple|None) - 生成的density prior boxes的大小，此属性应该为和 :attr:`densities` 有同样长度的列表或数组。默认值为None。
  - **fixed_ratios** (list|tuple|None) - 生成的density prior boxes的比值，如果该属性未被设置，同时 :attr:`densities` 和 :attr:`fix_sizes` 被设置，则 :attr:`aspect_ratios` 被用于生成 density prior boxes
  - **variance** (list|tuple) - 将被用于density prior boxes编码的方差，默认值为：[0.1, 0.1, 0.2, 0.2]
  - **clip** (bool) - 是否裁剪超出范围的box。默认值：False
  - **step** (list|tuple) - Prior boxes在宽度和高度的步长，如果step[0]等于0.0或step[1]等于0.0, input的the density prior boxes的高度/宽度的步长将被自动计算。默认值：Default: [0., 0.]
  - **offset** (float) - Prior boxes中心偏移值，默认为：0.5
  - **flatten_to_2d** (bool) - 是否将output prior boxes和方差 ``flatten`` 至2-D，其中第二个dim为4。默认值：False
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
含有两个变量的元组，包括：
  候选框：

    当flatten_to_2d为False时，形状为[H, W, num_priors, 4]的4-D Tensor。
    当flatten_to_2d为True时，形式为[H * W * num_priors, 4]的 2-D Tensor。
    其中，H是输入的高度，W是输入的宽度，num_priors是输入中每个位置的候选框数。

  候选框的方差：

    当flatten_to_2d为False时，形状为[H, W, num_priors, 4]的4-D Tensor。
    当flatten_to_2d为True时，形式为[H * W * num_priors, 4]的2-D Tensor。
    其中，H是输入的高度，W是输入的宽度，num_priors是输入中每个位置的候选框数。

返回类型
::::::::::::
元组

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.density_prior_box