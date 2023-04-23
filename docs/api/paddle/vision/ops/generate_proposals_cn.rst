.. _cn_api_paddle_vision_ops_generate_proposals:

generate_proposals
-------------------------------

.. py:function:: paddle.vision.ops.generate_proposals(scores, bbox_deltas, img_size, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, pixel_offset=False, return_rois_num=False, name=None)





根据每个检测框为 foreground 对象的概率，根据 ``anchors`` 和 ``bbox_deltas`` 以及 ``scores``  计算生成 RPN 的输出 proposal。最后的推选的 proposal 被用于训练检测网络。


该操作通过以下步骤生成 proposals ：

        1、通过转置操作将 ``scores`` 和 ``bbox_deltas`` 的大小分别调整为 ``（H * W * A，1）`` 和 ``（H * W * A，4）`` 。

        2. 计算出候选框的位置。

        3、将检测框的坐标限定到图像尺寸范围内。

        4、删除面积较小的候选框。

        5、通过非极大抑制(non-maximum suppression, NMS), 选出满足条件的候选框作为结果。

参数
::::::::::::
        - **scores** (Tensor) - Shape 为 ``[N，A，H，W]`` 的 4-D Tensor，表示每个框包含 object 的概率。N 是批大小，A 是 anchor 数，H 和 W 是 feature map 的高度和宽度。数据类型支持 float32。
        - **bbox_deltas** (Tensor)- Shape 为 ``[N，4 * A，H，W]`` 的 4-D Tensor，表示预测出的候选框的位置和 anchor 的位置之间的距离。数据类型支持 float32。
        - **img_size** (Tensor) - Shape 为 ``[N，2]`` 的 2-D Tensor，表示原始图像的大小信息。信息包含原始图像宽、高和 feature map 相对于原始图像缩放的比例。数据类型可为 float32 或 float64。
        - **anchors** (Tensor) - Shape 为 ``[H，W，A，4]`` 的 4-D Tensor。H 和 W 是 feature map 的高度和宽度。A 是每个位置的框的数量。每个 anchor 以 ``（xmin，ymin，xmax，ymax）`` 的格式表示，其中， ``xmin`` 和 ``ymin`` 为左上角的坐标， ``xmax`` 和 ``ymax`` 为右下角的坐标。数据类型支持 float32。
        - **variances** (Tensor) - Shape 为 ``[H，W，A，4]`` 的 4-D Tensor，表示 ``anchors`` 的方差。每个 anchor 的方差都是 ``（xcenter，ycenter，w，h）`` 的格式表示。数据类型支持 float32。
        - **pre_nms_top_n** (int，可选) - 每张图在 NMS 操作之前要保留的总框数。默认值为 6000。
        - **post_nms_top_n** (int，可选) - 每个图在 NMS 后要保留的总框数。默认值为 1000。
        - **nms_thresh** (float，可选) - NMS 中的阈值。默认值为 0.5。
        - **min_size** (float，可选) - 根据宽和高过滤候选框的阈值，宽或高小于该阈值的候选框将被过滤掉。默认值为 0.1。
        - **eta** (float，可选) - 自适应阈值的衰减系数。仅在自适应 NMS 中且自适应阈值大于 0.5 时生效，在每次迭代中 ``adaptive_threshold = adaptive_treshold * eta`` 。默认值为 1.0。
        - **pixel_offset** (bool, 可选）- 是否有像素偏移。如果是 True, ``img_size`` 在计算时会偏移 1。默认值为 False。
        - **return_rois_num** (bool，可选) - 是否返回 ``rpn_rois_num`` 。当设定为 True 时会返回一个形状为[N,]的 1-D 的 Tensor，包含该 Batch 中每一张图片包含的 RoI 的数目。 N 是批大小和图片数量。默认值为 False。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
- **rpn_rois** (Tensor) - 生成的 RoIs。为形状是 ``[N, 4]`` 的 2-D Tensor， 其中 N 为 RoIs 的数量。数据类型与 ``scores`` 一致。
- **rpn_roi_probs** (Tensor) - 生成的 RoIs 的得分。为形状是为 ``[N, 1]`` 的 2-D Tensor，其中 N 为 RoIs 的数量。数据类型与 ``scores`` 一致。
- **rpn_rois_num** (Tensor) - 该 Batch 中每一张图片包含的 RoI 的数目。为形状是为 ``[B,]`` 的 1-D Tensor。其中 ``B`` 是批大小和图片数量。此外，其和与 RoIs 的数量 ``N`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.vision.ops.generate_proposals
