.. _cn_api_fluid_layers_retinanet_detection_output:

retinanet_detection_output
-------------------------------

.. py:function:: paddle.fluid.layers.retinanet_detection_output(bboxes, scores, anchors, im_info, score_threshold=0.05, nms_top_k=1000, keep_top_k=100, nms_threshold=0.3, nms_eta=1.0)




在 `RetinaNet <https://arxiv.org/abs/1708.02002>`_ 中，有多个 `FPN <https://arxiv.org/abs/1612.03144>`_ 层会输出用于分类的预测值和位置回归的预测值，该OP通过执行以下步骤将这些预测值转换成最终的检测结果：

1. 在每个FPN层上，先剔除分类预测值小于score_threshold的anchor，然后按分类预测值从大到小排序，选出排名前nms_top_k的anchor，并将这些anchor与其位置回归的预测值做解码操作得到检测框。
2. 合并全部FPN层上的检测框，对这些检测框进行非极大值抑制操作（NMS）以获得最终的检测结果。


参数
::::::::::::

    - **bboxes**  (List) – 由来自不同FPN层的Tensor组成的列表，表示全部anchor的位置回归预测值。列表中每个元素是一个维度为 :math:`[N, Mi, 4]` 的3-D Tensor，其中，第一维N表示批量训练时批量内的图片数量，第二维Mi表示每张图片第i个FPN层上的anchor数量，第三维4表示每个anchor有四个坐标值。数据类型为float32或float64。
    - **scores**  (List) – 由来自不同FPN层的Tensor组成的列表，表示全部anchor的分类预测值。列表中每个元素是一个维度为 :math:`[N, Mi, C]` 的3-D Tensor，其中第一维N表示批量训练时批量内的图片数量，第二维Mi表示每张图片第i个FPN层上的anchor数量，第三维C表示类别数量（ **不包括背景类** ）。数据类型为float32或float64。
    - **anchors**  (List) – 由来自不同FPN层的Tensor组成的列表，表示全部anchor的坐标值。列表中每个元素是一个维度为 :math:`[Mi, 4]` 的2-D Tensor，其中第一维Mi表示第i个FPN层上的anchor数量，第二维4表示每个anchor有四个坐标值（[xmin, ymin, xmax, ymax]）。数据类型为float32或float64。
    - **im_info**  (Variable) – 维度为 :math:`[N, 3]` 的2-D Tensor，表示输入图片的尺寸信息。其中，第一维N表示批量训练时各批量内的图片数量，第二维3表示各图片的尺寸信息，分别是网络输入尺寸的高和宽，以及原图缩放至网络输入大小时的缩放比例。数据类型为float32或float64。
    - **score_threshold**  (float32) – 在NMS步骤之前，用于滤除每个FPN层的检测框的阈值，默认值为0.05。
    - **nms_top_k**  (int32) – 在NMS步骤之前，保留每个FPN层的检测框的数量，默认值为1000。
    - **keep_top_k**  (int32) – 在NMS步骤之后，每张图像要保留的检测框数量，默认值为100，若设为-1，则表示保留NMS步骤后剩下的全部检测框。
    - **nms_threshold**  (float32) – NMS步骤中用于剔除检测框的Intersection-over-Union（IoU）阈值，默认为0.3。
    - **nms_eta**  (float32) – NMS步骤中用于调整nms_threshold的参数。默认值为1.，表示nms_threshold的取值在NMS步骤中一直保持不变，即其设定值。若nms_eta小于1.，则表示当nms_threshold的取值大于0.5时，每保留一个检测框就调整一次nms_threshold的取值，即nms_threshold = nms_threshold * nms_eta，直到nms_threshold的取值小于等于0.5后结束调整。
**注意：在模型输入尺寸特别小的情况，此时若用score_threshold滤除anchor，可能会导致没有任何检测框剩余。为避免这种情况出现，该OP不会对最高FPN层上的anchor做滤除。因此，要求bboxes、scores、anchors中最后一个元素是来自最高FPN层的Tensor** 。

返回
::::::::::::
维度是 :math:`[No, 6]` 的2-D LoDTensor，表示批量内的检测结果。第一维No表示批量内的检测框的总数，第二维6表示每行有六个值：[label， score，xmin，ymin，xmax，ymax]。该LoDTensor的LoD中存放了每张图片的检测框数量，第i张图片的检测框数量为 :math:`LoD[i + 1] - LoD[i]`。如果 :math:`LoD[i + 1] - LoD[i]` 为0，则第i个图像没有检测结果。如果批量内的全部图像都没有检测结果，则LoD中所有元素被设置为0，LoDTensor被赋为空（None）。


返回类型
::::::::::::
变量（Variable），数据类型为float32或float64。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.retinanet_detection_output