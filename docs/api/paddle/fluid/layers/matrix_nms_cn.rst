.. _cn_api_fluid_layers_matrix_nms:

matrix_nms
-------------------------------


.. py:function:: paddle.fluid.layers.matrix_nms(bboxes, scores, score_threshold, post_threshold, nms_top_k, keep_top_k, use_gaussian=False, gaussian_sigma=2., background_label=0, normalized=True, return_index=False, name=None)




**Matrix NMS**

该 OP 使用 Matrix NMS 算法对边界框（bounding box）和评分（scores）执行多类非极大值抑制（NMS）。

如果提供 ``score_threshold`` 阈值且 ``nms_top_k`` 大于-1，则选择置信度分数最大的 k 个框。然后按照 Matrix NMS 算法对分数进行衰减。经过抑制后，如果 ``keep_top_k`` 大于-1，则每张图片最终保留 ``keep_top_k`` 个检测框。

在 NMS 步骤后，如果 keep_top_k 大于-1，则每个图像最多保留 keep_top_k 个框（bounding box）。


参数
::::::::::::

    - **bboxes**  (Variable) - 形为[N，M，4]的 3-DTensor，表示将预测 M 个边界框的预测位置，N 是批大小（batch size）。当边界框(bounding box)大小等于 4 时，每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。数据类型为 float32 或 float64。
    - **scores**  (Variable) – 形为[N，C，M]的 3-DTensor，表示预测的置信度。N 是批大小（batch size），C 是种类数目，M 是边界框 bounding box 的数量。对于每个类别，存在对应于 M 个边界框的总 M 个分数。请注意，M 等于 bboxes 的第二维。数据类型为 float32 或 float64。
    - **score_threshold**  (float) – 过滤掉低置信度分数的边界框的阈值。
    - **post_threshold**  (float) – 经过 NMS 衰减后，过滤掉低置信度分数的边界框的阈值。
    - **nms_top_k**  (int) – 基于 score_threshold 的过滤检测后，根据置信度保留的最大检测次数。
    - **keep_top_k**  (int) – 经过 NMS 抑制后，最终保留的最大检测次数。如果设置为 -1，则则保留全部。
    - **use_gaussian**  (bool) –  是否使用高斯函数衰减。默认值：False 。
    - **gaussian_sigma**  (float) – 高斯函数的 Sigma 值，默认值：2.0 。
    - **background_label**  (int) – 背景标签（类别）的索引，如果设置为 0，则忽略背景标签（类别）。如果设置为 -1，则考虑所有类别。默认值：0
    - **normalized**  (bool) –  检测是否已经经过正则化。默认值：True 。
    - **return_index**  (bool) –  是否同时返回保留检测框的序号。默认值：False 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - **Out**  (Variable) - 形为[No，6]的 2-D LoDTensor，表示检测结果。每行有 6 个值：[标签 label，置信度 confidence，xmin，ymin，xmax，ymax]。或形为[No，10]的 2-D LoDTensor，用来表示检测结果。每行有 10 个值：[标签 label，置信度 confidence，x1，y1，x2，y2，x3，y3，x4，y4]。 No 是检测的总数。如果对所有图像都没有检测到的 box，则 lod 将设置为{1}，而 Out 仅包含一个值-1。 （1.3 版本之后，当未检测到 box 时，lod 从{0}更改为{1}）
    - **Index**  (Variable) - 形为[No，1]的 2-D LoDTensor，表示检测结果在整个批次中的序号。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.matrix_nms
