.. _cn_api_fluid_layers_density_prior_box:

density_prior_box
-------------------------------

.. py:function:: paddle.fluid.layers.density_prior_box(input, image, densities=None, fixed_sizes=None, fixed_ratios=None, variance=[0.1, 0.1, 0.2, 0.2], clip=False, steps=[0.0, 0.0], offset=0.5, flatten_to_2d=False, name=None)


**Density Prior Box Operator**

为SSD算法(Single Shot MultiBox Detector)生成density prior box。
每个input的位置产生N个prior box，其中，N通过densities, fixed_sizes and fixed_ratios
的量来决定。在每个input位置附近的box center格点，通过此op生成。格点坐标由densities决定，
density prior box的量由fixed_sizes and fixed_ratios决定。显然地，fixed_sizes
和densities相等。对于densities中的densities_i：

.. math::

  N\_density\_prior\_box =sum(N\_fixed\_ratios * {densities\_i}^2)


参数：
  - **input** (Variable) - 输入变量，格式为NCHW
  - **image** (Variable) - PriorBoxOp的输入图像数据，格式为NCHW
  - **densities** (list|tuple|None) - 被生成的density prior boxes的densities，此属性应该是一个整数列表或数组。默认值为None
  - **fixed_sizes** (list|tuple|None) - 被生成的density prior boxes的固定大小，此属性应该为和 :attr:`densities` 有同样长度的列表或数组。默认值为None
  - **fixed_ratios** (list|tuple|None) - 被生成的density prior boxes的固定长度，如果该属性未被设置，同时 :attr:`densities` 和 :attr:`fix_sizes` 被设置，则 :attr:`aspect_ratios` 被用于生成 density prior boxes
  - **variance** (list|tuple) - 将被用于density prior boxes编码的方差，默认值为:[0.1, 0.1, 0.2, 0.2]
  - **clip(bool)** - 是否clip超出范围的box。默认值：False
  - **step** (list|turple) - Prior boxes在宽度和高度的步长，如果step[0] == 0.0/step[1] == 0.0, input的the density prior boxes的高度/宽度的步长将被自动计算。默认值：Default: [0., 0.]
  - **offset** (float) - Prior boxes中心补偿值，默认为：0.5
  - **flatten_to_2d** (bool) - 是否将output prior boxes和方差 ``flatten`` 至2维形状，第二个dim为4。默认值：False
  - **name(str)** - density prior box op的名字，默认值: None

返回：
  tuple: 有两个变量的数组 (boxes, variances)

  boxes: PriorBox的输出density prior boxes

    当flatten_to_2d为False时，形式为[H, W, num_priors, 4]

    当flatten_to_2d为True时，形式为[H * W * num_priors, 4]

    H是输入的高度，W是输入的宽度

    num_priors是输入中每个位置的总box count

  variances:  PriorBox的expanded variance

    当flatten_to_2d为False时，形式为[H, W, num_priors, 4]

    当flatten_to_2d为True时，形式为[H * W * num_priors, 4]

    H是输入的高度，W是输入的宽度

    num_priors是输入中每个位置的总box count

**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[3,6,9])
    images = fluid.layers.data(name="images", shape=[3,9,12])
    box, var = fluid.layers.density_prior_box(
        input=input,
        image=images,
        densities=[4, 2, 1],
        fixed_sizes=[32.0, 64.0, 128.0],
        fixed_ratios=[1.],
        clip=True,
        flatten_to_2d=True)











