.. _cn_api_fluid_layers_generate_proposals:

generate_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposals(scores, bbox_deltas, im_info, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, name=None)





该OP根据每个检测框为foreground对象的概率，推选生成用于后续检测网络的RoIs。
其中的检测框根据 ``anchors`` 和 ``bbox_deltas`` 计算得到。


该OP通过以下步骤生成 ``RoIs`` ：

        1、通过转置操作将 ``scores`` 和 ``bbox_deltas`` 的大小分别调整为 ``（H * W * A，1）`` 和 ``（H * W * A，4）`` 。

        2、根据 ``anchors`` 和 ``bbox_deltas`` 计算出候选框的位置。

        3、Clip boxes to image。

        4、删除面积较小的候选框。

        5、通过NMS选出满足条件的候选框作为结果。

参数
::::::::::::

        - **scores** (Variable) - Shape为 ``[N，A，H，W]`` 的4-D Tensor，表示每个框包含object的概率。N是批量大小，A是anchor数，H和W是feature map的高度和宽度。数据类型支持float32。
        - **bbox_deltas** (Variable)- Shape为 ``[N，4 * A，H，W]`` 的4-D Tensor，表示预测出的候选框的位置和anchor的位置之间的距离。数据类型支持float32。
        - **im_info** (Variable) - Shape为 ``[N，3]`` 的2-D张量，表示原始图像的大小信息。信息包含原始图像宽、高和feature map相对于原始图像缩放的比例。
        - **anchors** (Variable) - Shape为 ``[H，W，A，4]`` 的4-D Tensor。H和W是feature map的高度和宽度。A是每个位置的框的数量。每个anchor以 ``（xmin，ymin，xmax，ymax）`` 的格式表示，其中，``xmin`` 和 ``ymin`` 为左上角的坐标，``xmax`` 和 ``ymax`` 为右下角的坐标。
        - **variances** (Variable) - Shape为 ``[H，W，A，4]`` 的4-D Tensor，表示 ``anchors`` 的方差。每个anchor的方差都是 ``（xcenter，ycenter，w，h）`` 的格式表示。
        - **pre_nms_top_n** (int，可选) - 整型数字。每张图在NMS操作之前要保留的总框数。数据类型仅支持int32。缺省值为6000。
        - **post_nms_top_n** (int，可选) - 整型数字。每个图在NMS后要保留的总框数。数据类型仅支持int32。缺省值为1000。
        - **nms_thresh** (float，可选) - 浮点型数字。NMS中的阈值。数据类型仅支持float32。缺省值为0.5。
        - **min_size** (float，可选) - 浮点型数字。根据宽和高过滤候选框的阈值，宽或高小于该阈值的候选框将被过滤掉。数据类型仅支持float32。缺省值为0.1。
        - **eta** (float，可选) - 浮点型数字。自适应阈值的衰减系数。仅在自适应NMS中且自适应阈值大于0.5时生效，在每次迭代中 ``adaptive_threshold = adaptive_treshold * eta``。缺省值为1.0。


返回
::::::::::::
 元组，格式为 ``(rpn_rois, rpn_roi_probs)`` 

- **rpn_rois** (Variable) - 表示产出的RoIs, shape为 ``[N, 4]`` 的2D LoDTensor， N为RoIs的数量。数据类型与 ``scores`` 一致。
- **rpn_roi_probs** (Variable) - 表示RoIs的得分，shape为 ``[N, 1]`` ，N为RoIs的数量。数据类型与 ``scores`` 一致。

返回类型
::::::::::::
元组

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.generate_proposals