.. _cn_api_fluid_layers_matrix_nms:

matrix_nms
-------------------------------


.. py:function:: paddle.fluid.layers.matrix_nms(bboxes, scores, score_threshold, post_threshold, nms_top_k, keep_top_k, use_gaussian=False, gaussian_sigma=2., background_label=0, normalized=True, return_index=False, name=None)




**Matrix NMS**

该OP使用Matrix NMS算法对边界框（bounding box）和评分（scores）执行多类非极大值抑制（NMS）。

如果提供 ``score_threshold`` 阈值且 ``nms_top_k`` 大于-1，则选择置信度分数最大的k个框。然后按照Matrix NMS算法对分数进行衰减。经过抑制后，如果 ``keep_top_k`` 大于-1，则每张图片最终保留 ``keep_top_k`` 个检测框。

在NMS步骤后，如果keep_top_k大于-1，则每个图像最多保留keep_top_k个框（bounding box）。


参数
::::::::::::

    - **bboxes**  (Variable) - 形为[N，M，4]的3-D张量，表示将预测M个边界框的预测位置，N是批大小（batch size）。当边界框(bounding box)大小等于4时，每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。数据类型为float32或float64。
    - **scores**  (Variable) – 形为[N，C，M]的3-D张量，表示预测的置信度。N是批大小（batch size），C是种类数目，M是边界框bounding box的数量。对于每个类别，存在对应于M个边界框的总M个分数。请注意，M等于bboxes的第二维。数据类型为float32或float64。
    - **score_threshold**  (float) – 过滤掉低置信度分数的边界框的阈值。
    - **post_threshold**  (float) – 经过NMS衰减后，过滤掉低置信度分数的边界框的阈值。
    - **nms_top_k**  (int) – 基于 score_threshold 的过滤检测后，根据置信度保留的最大检测次数。
    - **keep_top_k**  (int) – 经过NMS抑制后，最终保留的最大检测次数。如果设置为 -1，则则保留全部。
    - **use_gaussian**  (bool) –  是否使用高斯函数衰减。默认值：False 。
    - **gaussian_sigma**  (float) – 高斯函数的Sigma值，默认值：2.0 。
    - **background_label**  (int) – 背景标签（类别）的索引，如果设置为 0，则忽略背景标签（类别）。如果设置为 -1，则考虑所有类别。默认值：0
    - **normalized**  (bool) –  检测是否已经经过正则化。默认值：True 。
    - **return_index**  (bool) –  是否同时返回保留检测框的序号。默认值：False 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - **Out**  (Variable) - 形为[No，6]的2-D LoDTensor，表示检测结果。每行有6个值：[标签label，置信度confidence，xmin，ymin，xmax，ymax]。或形为[No，10]的2-D LoDTensor，用来表示检测结果。每行有10个值：[标签label，置信度confidence，x1，y1，x2，y2，x3，y3，x4，y4]。 No是检测的总数。如果对所有图像都没有检测到的box，则lod将设置为{1}，而Out仅包含一个值-1。 （1.3版本之后，当未检测到box时，lod从{0}更改为{1}）
    - **Index**  (Variable) - 形为[No，1]的2-D LoDTensor，表示检测结果在整个批次中的序号。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.matrix_nms