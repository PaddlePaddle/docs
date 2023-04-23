.. _cn_api_fluid_layers_generate_proposals:

generate_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposals(scores, bbox_deltas, im_info, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, name=None)





该 OP 根据每个检测框为 foreground 对象的概率，推选生成用于后续检测网络的 RoIs。
其中的检测框根据 ``anchors`` 和 ``bbox_deltas`` 计算得到。


该 OP 通过以下步骤生成 ``RoIs`` ：

        1、通过转置操作将 ``scores`` 和 ``bbox_deltas`` 的大小分别调整为 ``（H * W * A，1）`` 和 ``（H * W * A，4）`` 。

        2、根据 ``anchors`` 和 ``bbox_deltas`` 计算出候选框的位置。

        3、Clip boxes to image。

        4、删除面积较小的候选框。

        5、通过 NMS 选出满足条件的候选框作为结果。

参数
::::::::::::

        - **scores** (Variable) - Shape 为 ``[N，A，H，W]`` 的 4-D Tensor，表示每个框包含 object 的概率。N 是批量大小，A 是 anchor 数，H 和 W 是 feature map 的高度和宽度。数据类型支持 float32。
        - **bbox_deltas** (Variable)- Shape 为 ``[N，4 * A，H，W]`` 的 4-D Tensor，表示预测出的候选框的位置和 anchor 的位置之间的距离。数据类型支持 float32。
        - **im_info** (Variable) - Shape 为 ``[N，3]`` 的 2-DTensor，表示原始图像的大小信息。信息包含原始图像宽、高和 feature map 相对于原始图像缩放的比例。
        - **anchors** (Variable) - Shape 为 ``[H，W，A，4]`` 的 4-D Tensor。H 和 W 是 feature map 的高度和宽度。A 是每个位置的框的数量。每个 anchor 以 ``（xmin，ymin，xmax，ymax）`` 的格式表示，其中，``xmin`` 和 ``ymin`` 为左上角的坐标，``xmax`` 和 ``ymax`` 为右下角的坐标。
        - **variances** (Variable) - Shape 为 ``[H，W，A，4]`` 的 4-D Tensor，表示 ``anchors`` 的方差。每个 anchor 的方差都是 ``（xcenter，ycenter，w，h）`` 的格式表示。
        - **pre_nms_top_n** (int，可选) - 整型数字。每张图在 NMS 操作之前要保留的总框数。数据类型仅支持 int32。缺省值为 6000。
        - **post_nms_top_n** (int，可选) - 整型数字。每个图在 NMS 后要保留的总框数。数据类型仅支持 int32。缺省值为 1000。
        - **nms_thresh** (float，可选) - 浮点型数字。NMS 中的阈值。数据类型仅支持 float32。缺省值为 0.5。
        - **min_size** (float，可选) - 浮点型数字。根据宽和高过滤候选框的阈值，宽或高小于该阈值的候选框将被过滤掉。数据类型仅支持 float32。缺省值为 0.1。
        - **eta** (float，可选) - 浮点型数字。自适应阈值的衰减系数。仅在自适应 NMS 中且自适应阈值大于 0.5 时生效，在每次迭代中 ``adaptive_threshold = adaptive_treshold * eta``。缺省值为 1.0。


返回
::::::::::::
 元组，格式为 ``(rpn_rois, rpn_roi_probs)``

- **rpn_rois** (Variable) - 表示产出的 RoIs, shape 为 ``[N, 4]`` 的 2D LoDTensor， N 为 RoIs 的数量。数据类型与 ``scores`` 一致。
- **rpn_roi_probs** (Variable) - 表示 RoIs 的得分，shape 为 ``[N, 1]`` ，N 为 RoIs 的数量。数据类型与 ``scores`` 一致。

返回类型
::::::::::::
元组

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.generate_proposals
