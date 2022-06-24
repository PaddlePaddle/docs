.. _cn_api_fluid_layers_rpn_target_assign:

rpn_target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.rpn_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info, rpn_batch_size_per_im=256, rpn_straddle_thresh=0.0, rpn_fg_fraction=0.5, rpn_positive_overlap=0.7, rpn_negative_overlap=0.3, use_random=True)




该OP用于为anchors分配分类标签和回归标签，以便用这些标签对RPN进行训练。

该OP将anchors分为两种类别，正和负。根据Faster-RCNN的paper，正类别anchor包括以下两种anchor:

- 在与一个ground-truth boxes相交的所有anchor中，IoU最高的anchor
- 和任意一个ground-truth box的IoU超出了阈值 ``rpn_positive_overlap``

负类别anchor是和任何ground-truth boxes的IoU都低于阈值 ``rpn_negative_overlap`` 的anchor。

正负anchors之外的anchors不会被选出来参与训练。

回归标签是ground-truth boxes和正类别anchor的偏移值。

参数
::::::::::::

        - **bbox_pred** (Variable) - Shape为 ``[batch_size，M，4]`` 的3-D Tensor，表示M个边界框的预测位置。每个边界框有四个坐标值，即 ``[xmin，ymin，xmax，ymax]``。数据类型支持float32和float64。
        - **cls_logits** (Variable)- Shape为 ``[batch_size，M，1]`` 的3-D Tensor，表示预测的置信度。1是frontground和background的sigmoid，M是边界框的数量。数据类型支持float32和float64。
        - **anchor_box** (Variable) - Shape为 ``[M，4]`` 的2-D Tensor，它拥有M个框，每个框可表示为 ``[xmin，ymin，xmax，ymax]`` ， ``[xmin，ymin]`` 是anchor框的左上部坐标，如果输入是图像特征图，则它们接近坐标系的原点。``[xmax，ymax]`` 是anchor框的右下部坐标。数据类型支持float32和float64。
        - **anchor_var** (Variable) - Shape为 ``[M，4]`` 的2-D Tensor，它拥有anchor的expand方差。数据类型支持float32和float64。
        - **gt_boxes** (Variable) - Shape为 ``[Ng，4]`` 的2-D LoDTensor， ``Ng`` 是一个batch内输入groundtruth boxes的总数。数据类型支持float32和float64。
        - **is_crowd** (Variable) –Shape为 ``[M, 1]`` 的2-D LoDTensor，M为groundtruth boxes的数量。用于标记boxes是否是crowd。数据类型支持int32。
        - **im_info** (Variable) - Shape为[N，3]的2-D张量，表示原始图像的大小信息。信息包含原始图像宽、高和feature map相对于原始图像缩放的比例。数据类型支持int32。
        - **rpn_batch_size_per_im** (int，可选) - 整型数字。每个图像中RPN示例总数。数据类型支持int32。缺省值为256。
        - **rpn_straddle_thresh** (float，可选) - 浮点数字。超出图像外部 ``straddle_thresh`` 个像素的RPN anchors会被删除。数据类型支持float32。缺省值为0.0。
        - **rpn_fg_fraction** (float，可选) - 浮点数字。标记为foreground boxes的数量占batch内总体boxes的比例。数据类型支持float32。缺省值为0.5。
        - **rpn_positive_overlap** (float，可选) - 浮点数字。和任意一个groundtruth box的 ``IoU`` 超出了阈值 ``rpn_positive_overlap`` 的box被判定为正类别。数据类型支持float32。缺省值为0.7。
        - **rpn_negative_overlap** (float，可选) - 浮点数字。负类别anchor是和任何ground-truth boxes的IoU都低于阈值 ``rpn_negative_overlap`` 的anchor。数据类型支持float32。缺省值为0.3。
        - **use_random** (bool，可选) – 布尔类型。是否使用随机采样来选择foreground boxes和background boxes。缺省值为True。

返回
::::::::::::
 元组。格式为 ``(predicted_scores, predicted_location, target_label, target_bbox, bbox_inside_weight)``
   - **predicted_scores** (Varible) - RPN预测的类别结果。Shape为 ``[F + B，1]`` 的2D Tensor。 ``F`` 为foreground anchor的数量，B为background anchor的数量。数据类型与 ``bbox_pred`` 一致。
   - **predicted_location** (Variable) - RPN预测的位置结果。Shape为 ``[F, 4]`` 的2D Tensor。数据类型与 ``bbox_pred`` 一致。
   - **target_label** (Variable) - Shape为 ``[F + B，1]`` 的2D Tensor。数据类型为int32。
   - **target_bbox** (Variable) - Shape为 ``[F, 4]`` 的2D Tensor。数据类型与 ``bbox_pred`` 一致。
   - **Bbox_inside_weight** (Variable) - Shape为 ``[F, 4]`` 的2D Tensor。数据类型与 ``bbox_pred`` 一致。

返回类型
::::::::::::
元组


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.rpn_target_assign