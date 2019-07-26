.. _cn_api_fluid_layers_generate_proposals:

generate_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposals(scores, bbox_deltas, im_info, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, name=None)

生成proposal的Faster-RCNN

该操作根据每个框为foreground（前景）对象的概率，并且通过anchors来计算这些框，进而提出RoI。Bbox_deltais和一个objects的分数作为是RPN的输出。最终 ``proposals`` 可用于训练检测网络。

为了生成 ``proposals`` ，此操作执行以下步骤：

        1、转置和调整bbox_deltas的分数和大小为（H * W * A，1）和（H * W * A，4）。

        2、计算方框位置作为 ``proposals`` 候选框。

        3、剪辑框图像。

        4、删除小面积的预测框。

        5、应用NMS以获得最终 ``proposals`` 作为输出。

参数：
        - **scores** (Variable)- 是一个shape为[N，A，H，W]的4-D张量，表示每个框成为object的概率。N是批量大小，A是anchor数，H和W是feature map的高度和宽度。
        - **bbox_deltas** （Variable）- 是一个shape为[N，4 * A，H，W]的4-D张量，表示预测框位置和anchor位置之间的差异。
        - **im_info** （Variable）- 是一个shape为[N，3]的2-D张量，表示N个批次原始图像的信息。信息包含原始图像大小和 ``feature map`` 的大小之间高度，宽度和比例。
        - **anchors** （Variable）- 是一个shape为[H，W，A，4]的4-D Tensor。H和W是 ``feature map`` 的高度和宽度，
        - **num_anchors** - 是每个位置的框的数量。每个anchor都是以非标准化格式（xmin，ymin，xmax，ymax）定义的。
        - **variances** （Variable）- anchor的方差，shape为[H，W，num_priors，4]。每个方差都是（xcenter，ycenter，w，h）这样的格式。
        - **pre_nms_top_n** （float）- 每个图在NMS之前要保留的总框数。默认为6000。
        - **post_nms_top_n** （float）- 每个图在NMS后要保留的总框数。默认为1000。
        - **nms_thresh** （float）- NMS中的阈值，默认为0.5。
        - **min_size** （float）- 删除高度或宽度小于min_size的预测框。默认为0.1。
        - **eta** （float）- 在自适应NMS中应用，如果自适应阈值> 0.5，则在每次迭代中使用adaptive_threshold = adaptive_treshold * eta。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    scores = fluid.layers.data(name='scores', shape=[2, 4, 5, 5],
                 append_batch_size=False, dtype='float32')
    bbox_deltas = fluid.layers.data(name='bbox_deltas', shape=[2, 16, 5, 5],
                 append_batch_size=False, dtype='float32')
    im_info = fluid.layers.data(name='im_info', shape=[2, 3],
                 append_batch_size=False, dtype='float32')
    anchors = fluid.layers.data(name='anchors', shape=[5, 5, 4, 4],
                 append_batch_size=False, dtype='float32')
    variances = fluid.layers.data(name='variances', shape=[5, 5, 10, 4],
                 append_batch_size=False, dtype='float32')
    rois, roi_probs = fluid.layers.generate_proposals(scores, bbox_deltas,
                 im_info, anchors, variances)









