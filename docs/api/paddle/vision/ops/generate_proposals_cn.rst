.. _cn_api_paddle_vision_ops_generate_proposals:

generate_proposals
-------------------------------

.. py:function:: paddle.vision.ops.generate_proposals(scores, bbox_deltas, img_size, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, pixel_offset=False, return_rois_num=False, name=None)





根据每个检测框为foreground对象的概率，根据 ``anchors`` 和 ``bbox_deltas`` 以及 ``scores``  计算生成RPN的输出proposal。最后的推选的proposal被用于训练检测网络。


该操作通过以下步骤生成proposals ：

        1、通过转置操作将 ``scores`` 和 ``bbox_deltas`` 的大小分别调整为 ``（H * W * A，1）`` 和 ``（H * W * A，4）`` 。

        2. 计算出候选框的位置。

        3、将检测框的坐标限定到图像尺寸范围内。

        4、删除面积较小的候选框。

        5、通过非极大抑制(non-maximum suppression, NMS), 选出满足条件的候选框作为结果。

参数
::::::::::::
        - **scores** (Tensor) - Shape为 ``[N，A，H，W]`` 的4-D Tensor，表示每个框包含object的概率。N是批大小，A是anchor数，H和W是feature map的高度和宽度。数据类型支持float32。
        - **bbox_deltas** (Tensor)- Shape为 ``[N，4 * A，H，W]`` 的4-D Tensor，表示预测出的候选框的位置和anchor的位置之间的距离。数据类型支持float32。
        - **img_size** (Tensor) - Shape为 ``[N，2]`` 的2-D张量，表示原始图像的大小信息。信息包含原始图像宽、高和feature map相对于原始图像缩放的比例。数据类型可为float32或float64。
        - **anchors** (Tensor) - Shape为 ``[H，W，A，4]`` 的4-D Tensor。H和W是feature map的高度和宽度。A是每个位置的框的数量。每个anchor以 ``（xmin，ymin，xmax，ymax）`` 的格式表示，其中， ``xmin`` 和 ``ymin`` 为左上角的坐标， ``xmax`` 和 ``ymax`` 为右下角的坐标。数据类型支持float32。
        - **variances** (Tensor) - Shape为 ``[H，W，A，4]`` 的4-D Tensor，表示 ``anchors`` 的方差。每个anchor的方差都是 ``（xcenter，ycenter，w，h）`` 的格式表示。数据类型支持float32。
        - **pre_nms_top_n** (int，可选) - 每张图在NMS操作之前要保留的总框数。默认值为6000。
        - **post_nms_top_n** (int，可选) - 每个图在NMS后要保留的总框数。默认值为1000。
        - **nms_thresh** (float，可选) - NMS中的阈值。默认值为0.5。
        - **min_size** (float，可选) - 根据宽和高过滤候选框的阈值，宽或高小于该阈值的候选框将被过滤掉。默认值为0.1。
        - **eta** (float，可选) - 自适应阈值的衰减系数。仅在自适应NMS中且自适应阈值大于0.5时生效，在每次迭代中 ``adaptive_threshold = adaptive_treshold * eta`` 。默认值为1.0。
        - **pixel_offset** (bool, 可选）- 是否有像素偏移。如果是True, ``img_size`` 在计算时会偏移1。默认值为False。
        - **return_rois_num** (bool，可选) - 是否返回 ``rpn_rois_num`` 。当设定为True时会返回一个形状为[N,]的1-D的Tensor，包含该Batch中每一张图片包含的RoI的数目。 N是批大小和图片数量。默认值为False。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
- **rpn_rois** (Tensor) - 生成的RoIs。为形状是 ``[N, 4]`` 的2-D Tensor， 其中N为RoIs的数量。数据类型与 ``scores`` 一致。
- **rpn_roi_probs** (Tensor) - 生成的RoIs的得分。为形状是为 ``[N, 1]`` 的2-D Tensor，其中N为RoIs的数量。数据类型与 ``scores`` 一致。
- **rpn_rois_num** (Tensor) - 该Batch中每一张图片包含的RoI的数目。为形状是为 ``[B,]`` 的1-D Tensor。其中 ``B`` 是批大小和图片数量。此外，其和与RoIs的数量 ``N`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.vision.ops.generate_proposals
