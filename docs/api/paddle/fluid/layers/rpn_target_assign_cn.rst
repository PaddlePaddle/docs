.. _cn_api_fluid_layers_rpn_target_assign:

rpn_target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.rpn_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info, rpn_batch_size_per_im=256, rpn_straddle_thresh=0.0, rpn_fg_fraction=0.5, rpn_positive_overlap=0.7, rpn_negative_overlap=0.3, use_random=True)




该 OP 用于为 anchors 分配分类标签和回归标签，以便用这些标签对 RPN 进行训练。

该 OP 将 anchors 分为两种类别，正和负。根据 Faster-RCNN 的 paper，正类别 anchor 包括以下两种 anchor:

- 在与一个 ground-truth boxes 相交的所有 anchor 中，IoU 最高的 anchor
- 和任意一个 ground-truth box 的 IoU 超出了阈值 ``rpn_positive_overlap``

负类别 anchor 是和任何 ground-truth boxes 的 IoU 都低于阈值 ``rpn_negative_overlap`` 的 anchor。

正负 anchors 之外的 anchors 不会被选出来参与训练。

回归标签是 ground-truth boxes 和正类别 anchor 的偏移值。

参数
::::::::::::

        - **bbox_pred** (Variable) - Shape 为 ``[batch_size，M，4]`` 的 3-D Tensor，表示 M 个边界框的预测位置。每个边界框有四个坐标值，即 ``[xmin，ymin，xmax，ymax]``。数据类型支持 float32 和 float64。
        - **cls_logits** (Variable)- Shape 为 ``[batch_size，M，1]`` 的 3-D Tensor，表示预测的置信度。1 是 frontground 和 background 的 sigmoid，M 是边界框的数量。数据类型支持 float32 和 float64。
        - **anchor_box** (Variable) - Shape 为 ``[M，4]`` 的 2-D Tensor，它拥有 M 个框，每个框可表示为 ``[xmin，ymin，xmax，ymax]`` ， ``[xmin，ymin]`` 是 anchor 框的左上部坐标，如果输入是图像特征图，则它们接近坐标系的原点。``[xmax，ymax]`` 是 anchor 框的右下部坐标。数据类型支持 float32 和 float64。
        - **anchor_var** (Variable) - Shape 为 ``[M，4]`` 的 2-D Tensor，它拥有 anchor 的 expand 方差。数据类型支持 float32 和 float64。
        - **gt_boxes** (Variable) - Shape 为 ``[Ng，4]`` 的 2-D LoDTensor， ``Ng`` 是一个 batch 内输入 groundtruth boxes 的总数。数据类型支持 float32 和 float64。
        - **is_crowd** (Variable) –Shape 为 ``[M, 1]`` 的 2-D LoDTensor，M 为 groundtruth boxes 的数量。用于标记 boxes 是否是 crowd。数据类型支持 int32。
        - **im_info** (Variable) - Shape 为[N，3]的 2-DTensor，表示原始图像的大小信息。信息包含原始图像宽、高和 feature map 相对于原始图像缩放的比例。数据类型支持 int32。
        - **rpn_batch_size_per_im** (int，可选) - 整型数字。每个图像中 RPN 示例总数。数据类型支持 int32。缺省值为 256。
        - **rpn_straddle_thresh** (float，可选) - 浮点数字。超出图像外部 ``straddle_thresh`` 个像素的 RPN anchors 会被删除。数据类型支持 float32。缺省值为 0.0。
        - **rpn_fg_fraction** (float，可选) - 浮点数字。标记为 foreground boxes 的数量占 batch 内总体 boxes 的比例。数据类型支持 float32。缺省值为 0.5。
        - **rpn_positive_overlap** (float，可选) - 浮点数字。和任意一个 groundtruth box 的 ``IoU`` 超出了阈值 ``rpn_positive_overlap`` 的 box 被判定为正类别。数据类型支持 float32。缺省值为 0.7。
        - **rpn_negative_overlap** (float，可选) - 浮点数字。负类别 anchor 是和任何 ground-truth boxes 的 IoU 都低于阈值 ``rpn_negative_overlap`` 的 anchor。数据类型支持 float32。缺省值为 0.3。
        - **use_random** (bool，可选) – 布尔类型。是否使用随机采样来选择 foreground boxes 和 background boxes。缺省值为 True。

返回
::::::::::::
 元组。格式为 ``(predicted_scores, predicted_location, target_label, target_bbox, bbox_inside_weight)``
   - **predicted_scores** (Varible) - RPN 预测的类别结果。Shape 为 ``[F + B，1]`` 的 2D Tensor。 ``F`` 为 foreground anchor 的数量，B 为 background anchor 的数量。数据类型与 ``bbox_pred`` 一致。
   - **predicted_location** (Variable) - RPN 预测的位置结果。Shape 为 ``[F, 4]`` 的 2D Tensor。数据类型与 ``bbox_pred`` 一致。
   - **target_label** (Variable) - Shape 为 ``[F + B，1]`` 的 2D Tensor。数据类型为 int32。
   - **target_bbox** (Variable) - Shape 为 ``[F, 4]`` 的 2D Tensor。数据类型与 ``bbox_pred`` 一致。
   - **Bbox_inside_weight** (Variable) - Shape 为 ``[F, 4]`` 的 2D Tensor。数据类型与 ``bbox_pred`` 一致。

返回类型
::::::::::::
元组


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.rpn_target_assign
