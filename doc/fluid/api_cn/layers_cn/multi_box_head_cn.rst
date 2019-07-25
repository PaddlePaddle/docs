.. _cn_api_fluid_layers_multi_box_head:

multi_box_head
-------------------------------

.. py:function:: paddle.fluid.layers.multi_box_head(inputs, image, base_size, num_classes, aspect_ratios, min_ratio=None, max_ratio=None, min_sizes=None, max_sizes=None, steps=None, step_w=None, step_h=None, offset=0.5, variance=[0.1, 0.1, 0.2, 0.2], flip=True, clip=False, kernel_size=1, pad=0, stride=1, name=None, min_max_aspect_ratios_order=False)

生成SSD（Single Shot MultiBox Detector）算法的候选框。有关此算法的详细信息，请参阅SSD论文 `SSD：Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_ 的2.2节。

参数：
        - **inputs** （list | tuple）- 输入变量列表，所有变量的格式为NCHW。
        - **image** （Variable）- PriorBoxOp的输入图像数据，布局为NCHW。
        - **base_size** （int）- base_size用于根据 ``min_ratio`` 和 ``max_ratio`` 来获取 ``min_size`` 和 ``max_size`` 。
        - **num_classes** （int）- 类的数量。
        - **aspect_ratios** （list | tuple）- 生成候选框的宽高比。 ``input`` 和 ``aspect_ratios`` 的长度必须相等。
        - **min_ratio** （int）- 生成候选框的最小比率。
        - **max_ratio** （int）- 生成候选框的最大比率。
        - **min_sizes** （list | tuple | None）- 如果len（输入）<= 2，则必须设置 ``min_sizes`` ，并且 ``min_sizes`` 的长度应等于输入的长度。默认值：无。
        - **max_sizes** （list | tuple | None）- 如果len（输入）<= 2，则必须设置 ``max_sizes`` ，并且 ``min_sizes`` 的长度应等于输入的长度。默认值：无。
        - **steps** （list | tuple）- 如果step_w和step_h相同，则step_w和step_h可以被steps替换。
        - **step_w** （list | tuple）- 候选框跨越宽度。如果step_w [i] == 0.0，将自动计算输跨越入[i]宽度。默认值：无。
        - **step_h** （list | tuple）- 候选框跨越高度，如果step_h [i] == 0.0，将自动计算跨越输入[i]高度。默认值：无。
        - **offset** （float）- 候选框中心偏移。默认值：0.5
        - **variance** （list | tuple）- 在候选框编码的方差。默认值：[0.1,0.1,0.2,0.2]。
        - **flip** （bool）- 是否翻转宽高比。默认值：false。
        - **clip** （bool）- 是否剪切超出边界的框。默认值：False。
        - **kernel_size** （int）- conv2d的内核大小。默认值：1。
        - **pad** （int | list | tuple）- conv2d的填充。默认值：0。
        - **stride** （int | list | tuple）- conv2d的步长。默认值：1，
        - **name** （str）- 候选框的名称。默认值：无。
        - **min_max_aspect_ratios_order** （bool）- 如果设置为True，则输出候选框的顺序为[min，max，aspect_ratios]，这与Caffe一致。请注意，此顺序会影响卷积层后面的权重顺序，但不会影响最终检测结果。默认值：False。

返回：一个带有四个变量的元组，（mbox_loc，mbox_conf，boxes, variances）:

    - **mbox_loc** ：预测框的输入位置。布局为[N，H * W * Priors，4]。其中 ``Priors`` 是每个输位置的预测框数。

    - **mbox_conf** ：预测框对输入的置信度。布局为[N，H * W * Priors，C]。其中 ``Priors`` 是每个输入位置的预测框数，C是类的数量。

    - **boxes** ： ``PriorBox`` 的输出候选框。布局是[num_priors，4]。 ``num_priors`` 是每个输入位置的总框数。

    - **variances** ： ``PriorBox`` 的方差。布局是[num_priors，4]。 ``num_priors`` 是每个输入位置的总窗口数。

返回类型：元组（tuple）

**代码示例**

..  code-block:: python
        
        import paddle.fluid as fluid
     
        images = fluid.layers.data(name='data', shape=[3, 300, 300], dtype='float32')
        conv1 = fluid.layers.data(name='conv1', shape=[512, 19, 19], dtype='float32')
        conv2 = fluid.layers.data(name='conv2', shape=[1024, 10, 10], dtype='float32')
        conv3 = fluid.layers.data(name='conv3', shape=[512, 5, 5], dtype='float32')
        conv4 = fluid.layers.data(name='conv4', shape=[256, 3, 3], dtype='float32')
        conv5 = fluid.layers.data(name='conv5', shape=[256, 2, 2], dtype='float32')
        conv6 = fluid.layers.data(name='conv6', shape=[128, 1, 1], dtype='float32')
        
        mbox_locs, mbox_confs, box, var = fluid.layers.multi_box_head(
          inputs=[conv1, conv2, conv3, conv4, conv5, conv6],
          image=images,
          num_classes=21,
          min_ratio=20,
          max_ratio=90,
          aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
          base_size=300,
          offset=0.5,
          flip=True,
          clip=True)




