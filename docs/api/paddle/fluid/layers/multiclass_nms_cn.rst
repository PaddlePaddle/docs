.. _cn_api_fluid_layers_multiclass_nms:

multiclass_nms
-------------------------------

.. py:function:: paddle.fluid.layers.multiclass_nms(bboxes, scores, score_threshold, nms_top_k, keep_top_k, nms_threshold=0.3, normalized=True, nms_eta=1.0, background_label=0, name=None)




**多分类NMS**

该OP用于对边界框（bounding box）和评分（scores）执行多类非极大值抑制（NMS）。

在NMS中，如果提供 ``score_threshold`` 阈值，则此OP贪心地选择所有得分（scores）高于 ``score_threshold`` 的检测边界框（bounding box）的子集，如果nms_top_k大于-1，则选择最大的nms_top_k置信度分数。接着，该OP依据 adaptive nms（基于 ``nms_threshold`` 和 ``nms_eta``），删除与已选择的框IOU(交并比)高于nms_threshold 的重叠框。

在NMS步骤后，如果keep_top_k大于-1，则每个图像最多保留keep_top_k个框（bounding box）。

计算示例：
    .. code-block:: text

        如果：
            box1.data = (2.0, 3.0, 7.0, 5.0)  数据格式为 (xmin, ymin, xmax, ymax)
            box1.scores = (0.7, 0.2, 0.4)   其中 (label0.score=0.7, label1.score=0.2, label2.cores=0.4)
            box2.data = (3.0, 4.0, 8.0, 5.0)
            box2.score = (0.3, 0.3, 0.1)
            nms_threshold = 0.3
            background_label = 0
            score_threshold = 0
        则：
            iou = 4/11 > 0.3
            out.data = [[1, 0.3, 3.0, 4.0, 8.0, 5.0],    
                         [2, 0.4, 2.0, 3.0, 7.0, 5.0]]
                         
            输出数据格式为 (label, confidence, xmin, ymin, xmax, ymax)



参数
::::::::::::

    - **bboxes**  (Variable) – 支持两种类型的边界框（bounding box）:

      1. （Tensor）形为[N，M，4 或 8、16、24、32]的3-D张量，表示将预测M个边界框的预测位置，N是批大小（batch size）。当边界框(bounding box)大小等于4时，每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。数据类型为float32或float64。
      2. （LoDTensor）形状为[M，C，4] M是边界框的个数，C是类别个数。数据类型为float32或float64

    - **scores**  (Variable) – 支持两种类型的分数：

      1. （Tensor）具有形状[N，C，M]的3-D张量表示预测的置信度。N是批量大小 batch size，C是种类数目，M是边界框bounding box的数量。对于每个类别，存在对应于M个边界框的总M个分数。请注意，M等于bboxes的第二维。数据类型为float32或float64。
      2. （LoDTensor）具有形状[M，C]的2-D LoDTensor。 M是bbox的数量，C是种类数目。在这种情况下，输入bboxes应该是形为[M，C，4]的第二种情况。数据类型为float32或float64。

    - **background_label**  (int) – 背景标签（类别）的索引，如果设置为 0，则忽略背景标签（类别）。如果设置为 -1，则考虑所有类别。默认值：0
    - **score_threshold**  (float) – 过滤掉低置信度分数的边界框的阈值。如果没有提供，请考虑所有边界框。
    - **nms_top_k**  (int) – 基于 score_threshold 的过滤检测后，根据置信度保留的最大检测次数。
    - **nms_threshold**  (float) – 在NMS中用于剔除检测框IOU的阈值，默认值：0.3 。
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

..  code-block:: python

    import paddle.fluid as fluid
    boxes = fluid.layers.data(name='bboxes', shape=[81, 4],
                              dtype='float32', lod_level=1)
    scores = fluid.layers.data(name='scores', shape=[81],
                              dtype='float32', lod_level=1)
    out = fluid.layers.multiclass_nms(bboxes=boxes,
                                      scores=scores,
                                      background_label=0,
                                      score_threshold=0.5,
                                      nms_top_k=400,
                                      nms_threshold=0.3,
                                      keep_top_k=200,
                                      normalized=False)



