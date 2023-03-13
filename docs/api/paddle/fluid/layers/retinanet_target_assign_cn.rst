.. _cn_api_fluid_layers_retinanet_target_assign:

retinanet_target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.retinanet_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, gt_labels, is_crowd, im_info, num_classes=1, positive_overlap=0.5, negative_overlap=0.4)




该 OP 是从输入 anchor 中找出训练检测模型 `RetinaNet <https://arxiv.org/abs/1708.02002>`_ 所需的正负样本，并为每个正负样本分配用于分类的目标值和位置回归的目标值，同时从全部 anchor 的类别预测值 cls_logits、位置预测值 bbox_pred 中取出属于各正负样本的部分。

正负样本的查找准则如下：
    - 若 anchor 与某个真值框之间的 Intersection-over-Union（IoU）大于其他 anchor 与该真值框的 IoU，则该 anchor 是正样本，且被分配给该真值框；
    - 若 anchor 与某个真值框之间的 IoU 大于等于 positive_overlap，则该 anchor 是正样本，且被分配给该真值框；
    - 若 anchor 与某个真值框之间的 IoU 介于[0, negative_overlap)，则该 anchor 是负样本；
    - 不满足以上准则的 anchor 不参与模型训练。

在 RetinaNet 中，对于每个 anchor，模型都会预测一个 C 维的向量用于分类，和一个 4 维的向量用于位置回归，因此各正负样本的分类目标值也是一个 C 维向量，各正样本的位置回归目标值也是一个 4 维向量。对于正样本而言，若其被分配的真值框的类别是 i，则其分类目标值的第 i-1 维为 1，其余维度为 0；其位置回归的目标值由 anchor 和真值框之间位置差值计算得到。对于负样本而言，其分类目标值的所有维度都为 0，因负样本不参与位置回归的训练，故负样本无位置回归的目标值。

分配结束后，从全部 anchor 的类别预测值 cls_logits 中取出属于各正负样本的部分，从针对全部 anchor 的位置预测值 bbox_pred 中取出属于各正样本的部分。


参数
::::::::::::

    - **bbox_pred**  (Variable) – 维度为 :math:`[N, M, 4]` 的 3-D Tensor，表示全部 anchor 的位置回归预测值。其中，第一维 N 表示批量训练时批量内的图片数量，第二维 M 表示每张图片的全部 anchor 的数量，第三维 4 表示每个 anchor 有四个坐标值。数据类型为 float32 或 float64。
    - **cls_logits**  (Variable) – 维度为 :math:`[N, M, C]` 的 3-D Tensor，表示全部 anchor 的分类预测值。其中，第一维 N 表示批量训练时批量内的图片数量，第二维 M 表示每张图片的全部 anchor 的数量，第三维 C 表示每个 anchor 需预测的类别数量（ **注意：不包括背景** ）。数据类型为 float32 或 float64。

    - **anchor_box**  (Variable) – 维度为 :math:`[M, 4]` 的 2-D Tensor，表示全部 anchor 的坐标值。其中，第一维 M 表示每张图片的全部 anchor 的数量，第二维 4 表示每个 anchor 有四个坐标值 :math:`[xmin, ymin, xmax, ymax]` ，:math:`[xmin, ymin]` 是 anchor 的左上顶部坐标，:math:`[xmax, ymax]` 是 anchor 的右下坐标。数据类型为 float32 或 float64。anchor_box 的生成请参考 OP :ref:`cn_api_fluid_layers_anchor_generator`。
    - **anchor_var**  (Variable) – 维度为 :math:`[M, 4]` 的 2-D Tensor，表示在后续计算损失函数时 anchor 坐标值的缩放比例。其中，第一维 M 表示每张图片的全部 anchor 的数量，第二维 4 表示每个 anchor 有四个坐标缩放因子。数据类型为 float32 或 float64。anchor_var 的生成请参考 OP :ref:`cn_api_fluid_layers_anchor_generator`。
    - **gt_boxes**  (Variable) – 维度为 :math:`[G, 4]` 且 LoD level 必须为 1 的 2-D LoDTensor，表示批量训练时批量内的真值框位置。其中，第一维 G 表示批量内真值框的总数，第二维表示每个真值框有四个坐标值。数据类型为 float32 或 float64。
    - **gt_labels**  (variable) – 维度为 :math:`[G, 1]` 且 LoD level 必须为 1 的 2-D LoDTensor，表示批量训练时批量内的真值框类别，数值范围为 :math:`[1, C]`。其中，第一维 G 表示批量内真值框的总数，第二维表示每个真值框只有 1 个类别。数据类型为 int32。
    - **is_crowd**  (Variable) – 维度为 :math:`[G]` 且 LoD level 必须为 1 的 1-D LoDTensor，表示各真值框是否位于重叠区域，值为 1 表示重叠，则不参与训练。第一维 G 表示批量内真值框的总数。数据类型为 int32。
    - **im_info**  (Variable) – 维度为 :math:`[N, 3]` 的 2-D Tensor，表示输入图片的尺寸信息。其中，第一维 N 表示批量训练时批量内的图片数量，第二维 3 表示各图片的尺寸信息，分别是网络输入尺寸的高和宽，以及原图缩放至网络输入尺寸的缩放比例。数据类型为 float32 或 float64。
    - **num_classes**  (int32) – 分类的类别数量，默认值为 1。
    - **positive_overlap**  (float32) – 判定 anchor 是一个正样本时 anchor 和真值框之间的最小 IoU，默认值为 0.5。
    - **negative_overlap**  (float32) – 判定 anchor 是一个负样本时 anchor 和真值框之间的最大 IoU，默认值为 0.4。该参数的设定值应小于等于 positive_overlap 的设定值，若大于，则 positive_overlap 的取值为 negative_overlap 的设定值。


返回
::::::::::::

    - **predict_scores** (Variable) – 维度为 :math:`[F + B, C]` 的 2-D Tensor，表示正负样本的分类预测值。其中，第一维 F 为批量内正样本的数量，B 为批量内负样本的数量，第二维 C 为分类的类别数量。数据类型为 float32 或 float64。
    - **predict_location** (Variable) — 维度为 :math:`[F, 4]` 的 2-D Tensor，表示正样本的位置回归预测值。其中，第一维 F 为批量内正样本的数量，第二维 4 表示每个样本有 4 个坐标值。数据类型为 float32 或 float64。
    - **target_label** (Variable) — 维度为 :math:`[F + B, 1]` 的 2-D Tensor，表示正负样本的分类目标值。其中，第一维 F 为正样本的数量，B 为负样本的数量，第二维 1 表示每个样本的真值类别只有 1 类。数据类型为 int32。
    - **target_bbox** (Variable) — 维度为 :math:`[F, 4]` 的 2-D Tensor，表示正样本的位置回归目标值。其中，第一维 F 为正样本的数量，第二维 4 表示每个样本有 4 个坐标值。数据类型为 float32 或 float64。
    - **bbox_inside_weight** (Variable) — 维度为 :math:`[F, 4]` 的 2-D Tensor，表示位置回归预测值中是否属于假正样本，若某个正样本为假，则 bbox_inside_weight 中对应维度的值为 0，否则为 1。第一维 F 为正样本的数量，第二维 4 表示每个样本有 4 个坐标值。数据类型为 float32 或 float64。
    - **fg_num** (Variable) — 维度为 :math:`[N, 1]` 的 2-D Tensor，表示正样本的数量。其中，第一维 N 表示批量内的图片数量。**注意：由于正样本数量会用作后续损失函数的分母，为避免出现除以 0 的情况，该 OP 已将每张图片的正样本数量做加 1 操作**。数据类型为 int32。


返回类型
::::::::::::
元组(tuple)，元组中的元素 predict_scores，predict_location，target_label，target_bbox，bbox_inside_weight，fg_num 都是 Variable。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.retinanet_target_assign
