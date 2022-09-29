.. _cn_api_paddle_vision_ops_prior_box:

prior_box
-------------------------------

.. py:function:: paddle.vision.ops.prior_box(input, image, min_sizes, max_sizes=None, aspect_ratios=[1.], variance=[0.1, 0.1, 0.2, 0.2], flip=False, clip=False, steps=[0.0, 0.0], offset=0.5, min_max_aspect_ratios_order=False, name=None)


为SSD(Single Shot MultiBox Detector)目标检测算法生成候选框，是在输入的每个位置生成 `N` 个候选框，`N` 由 `min_sizes`, `max_sizes` 和 `aspect_ratios` 的数目决定，候选框的尺寸在 `(min_size, max_size)` 之间，该尺寸根据 `aspect_ratios` 在序列中生成。


参数
::::::::::::
        - **input** (Tensor) - 形状为 `NCHW` 的4-D Tensor，数据类型为 float32 或 float64。
        - **image** (Tensor) - 输入图像数据，形状为 `NCHW` 的4-D Tensor，数据类型为 float32 或 float64。
        - **min_sizes** (list|tuple|float) - 生成的候选框的最小尺寸。
        - **max_sizes** (list|tuple|None) - 生成的候选框的最大尺寸。默认值为 None。
        - **aspect_ratios** (list|tuple|float) - 生成的候选框的长宽比。默认值为 [1.]。
        - **variance** (list|tuple) - 在候选框中解码的方差。默认值为 [0.1,0.1,0.2,0.2]。
        - **flip** (bool，可选) - 是否翻转。默认值为 False。
        - **clip** (bool，可选) - 是否裁剪。默认值为 False。
        - **step** (list|tuple) - 候选框在width和height上的步长。如果step[0]等于0.0或者step[1]等于0.0，则自动计算候选框在宽度和高度上的步长。默认值为 [0.,0.]。
        - **offset** (float) - 候选框中心位移。默认值为 0.5。
        - **min_max_aspect_ratios_order** (bool) - 若设为True，候选框的输出以[min, max, aspect_ratios]的顺序输出，和Caffe保持一致。请注意，该顺序会影响后面卷基层的权重顺序，但不影响最后的检测结果。默认值为 False。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
- **box** (Tensor) - 候选框，形状为 `[H,W,num_priors,4]` 的4-D Tensor。其中，H 是输入的高度，W 是输入的宽度，num_priors 是输入每个位置的总框数。
- **var** (Tensor) - 候选框的方差，形状为 `[H,W,num_priors,4]` 的4-D Tensor。其中，H是输入的高度，W是输入的宽度，num_priors 是输入每个位置的总框数。


代码示例
::::::::::::

COPY-FROM: paddle.vision.ops.prior_box
