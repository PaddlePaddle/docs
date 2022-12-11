.. _cn_api_fluid_layers_iou_similarity:

iou_similarity
-------------------------------

.. py:function:: paddle.fluid.layers.iou_similarity(x, y, box_normalized=True, name=None)




**IOU Similarity Operator**

计算两个框列表的 intersection-over-union(IOU)。框列表 :math:`X` 应为 LoDTensor， :math:`Y` 是普通 Tensor，:math:`X` 成批输入的所有实例共享 :math:`Y` 中的框。给定框 A 和框 B，IOU 的运算如下：

.. math::
    IOU(A, B) = \frac{area(A\cap B)}{area(A)+area(B)-area(A\cap B)}

参数
::::::::::::

    - **x** (Variable) - 框列表 :math:`X` 是二维 LoDTensor，维度为 :math:`[N,4]`，存有 :math:`N` 个框，每个框表示为 :math:`[xmin, ymin, xmax, ymax]` ，:math:`X` 的维度为 :math:`[N,4]`。如果输入是图像特征图，:math:`[xmin, ymin]` 表示框的左上角坐标，接近坐标轴的原点。:math:`[xmax, ymax]` 表示框的右下角坐标。该 Tensor 包含批次输入的 LoD 信息。该批次输入的一个实例能容纳不同的项数。数据类型为 float32 或 float64。
    - **y** (Variable) - 框列表 :math:`Y` 是二维 Tensor，存有 :math:`M` 个框，每个框表示为 :math:`[xmin, ymin, xmax, ymax]` ，:math:`Y` 的维度为 :math:`[M,4]`。如果输入是图像特征图，:math:`[xmin, ymin]` 表示框的左上角坐标，接近坐标轴的原点。:math:`[xmax, ymax]` 表示框的右下角坐标。数据类型为 float32 或 float64。
    - **box_normalized** (bool) - 先验框坐标是否正则化，即是否在[0, 1]区间内。默认值为 true

返回
::::::::::::
维度为 :math:`[N,M]` 的 LoDTensor，代表每一对 iou 分数，数据类型与 :math:`X` 相同

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.iou_similarity
