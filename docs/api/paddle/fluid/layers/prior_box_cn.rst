.. _cn_api_fluid_layers_prior_box:

prior_box
-------------------------------
.. py:function:: paddle.fluid.layers.prior_box(input,image,min_sizes=None,max_sizes=None,aspect_ratios=[1.0],variance=[0.1,0.1,0.2,0.2],flip=False,clip=False,steps=[0.0,0.0],offset=0.5,name=None,min_max_aspect_ratios_order=False)




该OP为SSD(Single Shot MultiBox Detector)算法生成候选框。输入的每个位产生N个候选框，N由min_sizes,max_sizes和aspect_ratios的数目决定，候选框的尺寸在(min_size,max_size)之间，该尺寸根据aspect_ratios在序列中生成。

参数
::::::::::::

    - **input** (Variable) - 形状为NCHW的4-DTensor，数据类型为float32或float64。
    - **image** (Variable) - PriorBoxOp的输入图像数据，形状为NCHW的4-D Tensor，数据类型为float32或float64。
    - **min_sizes** (list|tuple|float) - 生成的候选框的最小尺寸。
    - **max_sizes** (list|tuple|None) - 生成的候选框的最大尺寸。默认值为None
    - **aspect_ratios** (list|tuple|float) - 生成的候选框的长宽比。默认值为[1.]。
    - **variance** (list|tuple) - 在候选框中解码的方差。默认值为[0.1,0.1,0.2,0.2]。
    - **flip** (bool) - 是否翻转。默认值为False。
    - **clip** (bool) - 是否裁剪。默认值为False。
    - **step** (list|tuple) - 候选框在width和height上的步长。如果step[0]等于0.0或者step[1]等于0.0，则自动计算候选框在宽度和高度上的步长。默认：[0.,0.]
    - **offset** (float) - 候选框中心位移。默认：0.5
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **min_max_aspect_ratios_order** (bool) - 若设为True，候选框的输出以[min, max, aspect_ratios]的顺序输出，和Caffe保持一致。请注意，该顺序会影响后面卷基层的权重顺序，但不影响最后的检测结果。默认：False。

返回
::::::::::::
含有两个变量的元组，包括：
    boxes：候选框。形状为[H,W,num_priors,4]的4-D Tensor。其中，H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数。
    variances：候选框的方差，形状为[H,W,num_priors,4]的4-D Tensor。其中，H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数。

返回类型
::::::::::::
元组

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.prior_box