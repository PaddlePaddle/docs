.. _cn_api_fluid_layers_retinanet_detection_output:

retinanet_detection_output
-------------------------------

.. py:function:: paddle.fluid.layers.retinanet_detection_output(bboxes, scores, anchors, im_info, score_threshold=0.05, nms_top_k=1000, keep_top_k=100, nms_threshold=0.3, nms_eta=1.0)




在 `RetinaNet <https://arxiv.org/abs/1708.02002>`_ 中，有多个 `FPN <https://arxiv.org/abs/1612.03144>`_ 层会输出用于分类的预测值和位置回归的预测值，该 OP 通过执行以下步骤将这些预测值转换成最终的检测结果：

1. 在每个 FPN 层上，先剔除分类预测值小于 score_threshold 的 anchor，然后按分类预测值从大到小排序，选出排名前 nms_top_k 的 anchor，并将这些 anchor 与其位置回归的预测值做解码操作得到检测框。
2. 合并全部 FPN 层上的检测框，对这些检测框进行非极大值抑制操作（NMS）以获得最终的检测结果。


参数
::::::::::::

    - **bboxes**  (List) – 由来自不同 FPN 层的 Tensor 组成的列表，表示全部 anchor 的位置回归预测值。列表中每个元素是一个维度为 :math:`[N, Mi, 4]` 的 3-D Tensor，其中，第一维 N 表示批量训练时批量内的图片数量，第二维 Mi 表示每张图片第 i 个 FPN 层上的 anchor 数量，第三维 4 表示每个 anchor 有四个坐标值。数据类型为 float32 或 float64。
    - **scores**  (List) – 由来自不同 FPN 层的 Tensor 组成的列表，表示全部 anchor 的分类预测值。列表中每个元素是一个维度为 :math:`[N, Mi, C]` 的 3-D Tensor，其中第一维 N 表示批量训练时批量内的图片数量，第二维 Mi 表示每张图片第 i 个 FPN 层上的 anchor 数量，第三维 C 表示类别数量（ **不包括背景类** ）。数据类型为 float32 或 float64。
    - **anchors**  (List) – 由来自不同 FPN 层的 Tensor 组成的列表，表示全部 anchor 的坐标值。列表中每个元素是一个维度为 :math:`[Mi, 4]` 的 2-D Tensor，其中第一维 Mi 表示第 i 个 FPN 层上的 anchor 数量，第二维 4 表示每个 anchor 有四个坐标值（[xmin, ymin, xmax, ymax]）。数据类型为 float32 或 float64。
    - **im_info**  (Variable) – 维度为 :math:`[N, 3]` 的 2-D Tensor，表示输入图片的尺寸信息。其中，第一维 N 表示批量训练时各批量内的图片数量，第二维 3 表示各图片的尺寸信息，分别是网络输入尺寸的高和宽，以及原图缩放至网络输入大小时的缩放比例。数据类型为 float32 或 float64。
    - **score_threshold**  (float32) – 在 NMS 步骤之前，用于滤除每个 FPN 层的检测框的阈值，默认值为 0.05。
    - **nms_top_k**  (int32) – 在 NMS 步骤之前，保留每个 FPN 层的检测框的数量，默认值为 1000。
    - **keep_top_k**  (int32) – 在 NMS 步骤之后，每张图像要保留的检测框数量，默认值为 100，若设为-1，则表示保留 NMS 步骤后剩下的全部检测框。
    - **nms_threshold**  (float32) – NMS 步骤中用于剔除检测框的 Intersection-over-Union（IoU）阈值，默认为 0.3。
    - **nms_eta**  (float32) – NMS 步骤中用于调整 nms_threshold 的参数。默认值为 1.，表示 nms_threshold 的取值在 NMS 步骤中一直保持不变，即其设定值。若 nms_eta 小于 1.，则表示当 nms_threshold 的取值大于 0.5 时，每保留一个检测框就调整一次 nms_threshold 的取值，即 nms_threshold = nms_threshold * nms_eta，直到 nms_threshold 的取值小于等于 0.5 后结束调整。
**注意：在模型输入尺寸特别小的情况，此时若用 score_threshold 滤除 anchor，可能会导致没有任何检测框剩余。为避免这种情况出现，该 OP 不会对最高 FPN 层上的 anchor 做滤除。因此，要求 bboxes、scores、anchors 中最后一个元素是来自最高 FPN 层的 Tensor** 。

返回
::::::::::::
维度是 :math:`[No, 6]` 的 2-D LoDTensor，表示批量内的检测结果。第一维 No 表示批量内的检测框的总数，第二维 6 表示每行有六个值：[label， score，xmin，ymin，xmax，ymax]。该 LoDTensor 的 LoD 中存放了每张图片的检测框数量，第 i 张图片的检测框数量为 :math:`LoD[i + 1] - LoD[i]`。如果 :math:`LoD[i + 1] - LoD[i]` 为 0，则第 i 个图像没有检测结果。如果批量内的全部图像都没有检测结果，则 LoD 中所有元素被设置为 0，LoDTensor 被赋为空（None）。


返回类型
::::::::::::
变量（Variable），数据类型为 float32 或 float64。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.retinanet_detection_output
