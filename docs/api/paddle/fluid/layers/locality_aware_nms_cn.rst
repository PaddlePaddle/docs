.. _cn_api_fluid_layers_locality_aware_nms:

locality_aware_nms
-------------------------------

.. py:function:: paddle.fluid.layers.locality_aware_nms(bboxes, scores, score_threshold, nms_top_k, keep_top_k, nms_threshold=0.3, normalized=True, nms_eta=1.0, background_label=-1, name=None)




**局部感知NMS**

`局部感知NMS <https://arxiv.org/abs/1704.03155>`_ 用于对边界框（bounding box）和评分（scores）执行局部感知非极大值抑制（LANMS）。

首先，根据边界框之间的IOU(交并比)，对边界框和评分进行融合。

在NMS中，如果提供 ``score_threshold`` 阈值，则此OP贪心地选择所有得分（scores）高于 ``score_threshold`` 的检测边界框（bounding box）的子集，如果nms_top_k大于-1，则选择最大的nms_top_k置信度分数。接着，该OP依据 adaptive nms（基于 ``nms_threshold`` 和 ``nms_eta``），删除与已选择的框IOU(交并比)高于nms_threshold 的重叠框。

在NMS步骤后，如果keep_top_k大于-1，则每个图像最多保留keep_top_k个框（bounding box）。



参数
::::::::::::

    - **bboxes**  (Variable) – 支持两种类型的边界框（bounding box）:

      1. （Tensor）形为[N，M，4 或 8、16、24、32]的3-D张量，表示将预测M个边界框的预测位置，N是批大小（batch size）。当边界框(bounding box)大小等于4时，每个边界框有四个坐标值，布局为 :math:`[xmin, ymin, xmax, ymax]`。数据类型为float32或float64。

    - **scores**  (Variable) – 支持两种类型的分数：

      1. （Tensor）具有形状 :math:`[N, C, M]` 的3-D张量表示预测的置信度。N是批量大小 batch size，C是种类数目，M是边界框bounding box的数量。目前仅支持单个类别，所以输入维度应为 :math:`[N, 1, M]`。请注意，M等于bboxes的第二维。数据类型为float32或float64。

    - **background_label**  (int) – 背景标签（类别）的索引，如果设置为 0，则忽略背景标签（类别）。如果设置为 -1，则考虑所有类别。默认值：-1
    - **score_threshold**  (float) – 过滤掉低置信度分数的边界框的阈值。如果没有提供，请考虑所有边界框。
    - **nms_top_k**  (int) – 基于 score_threshold 的过滤检测后，根据置信度保留的最大检测次数。
    - **nms_threshold**  (float) – 在LANMS中用于融合检测框和剔除检测框IOU的阈值，默认值：0.3 。
    - **nms_eta**  (float) – 在NMS中用于调整 nms_threshold 的参数，设为1时表示nms_threshold不变。默认值：1.0 。
    - **keep_top_k**  (int) – NMS步骤后每个图像要保留的总bbox数。-1表示在NMS步骤之后保留所有bbox。
    - **normalized**  (bool) –  检测是否已经经过正则化。默认值：True 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
形为[No，6]的2-D LoDTensor，表示检测(detections)结果。每行有6个值：[标签label，置信度confidence，xmin，ymin，xmax，ymax]。或形为[No，10]的2-D LoDTensor，用来表示检测结果。每行有10个值：[标签label，置信度confidence，x1，y1，x2，y2，x3，y3，x4，y4]。 No是检测的总数。如果对所有图像都没有检测到的box，则lod将设置为{1}，而Out仅包含一个值-1。 （1.3版本之后，当未检测到box时，lod从{0}更改为{1}）

返回类型
::::::::::::
Variable，数据类型与输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.locality_aware_nms