.. _cn_api_paddle_vision_ops_nms:

nms
-------------------------------

.. py:function:: paddle.vision.ops.nms(boxes, iou_threshold=0.3, scores=None, category_idxs=None, categories=None, top_k=None)

非极大抑制(non-maximum suppression, NMS)用于在目标检测应用对检测边界框(bounding box)中搜索局部最大值，即只保留处于同一检测目标位置处重叠的框中分数最大的一个框。IoU(Intersection Over Union) 被用于判断两个框是否重叠，该值大于门限值(iou_threshold)则被认为两个框重叠。其计算公式如下：

.. math::

    IoU = \frac{intersection\_area(box1, box2)}{union\_area(box1, box2)}

如果参数 scores 不为 None，输入的 boxes 会首先按照它们对应的 score 降序排序，否则将默认输入的 boxes 为排好序的。

如果 category_idxs 和 categories 不为 None，分类 NMS 将会被执行，也就是说，nms 过程会在每一个类别的框当中分别进行计算，计算结果会被组合起来然后按照得分倒序排列。

如果 top_k 不为 None 的话，排序的计算结果中仅有前 k 个元素会被返回，否则会返回所有的元素。

参数
:::::::::
    - **boxes** (Tensor) - 待进行计算的框坐标，它应当是一个形状为[num_boxes, 4]的 2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出，数据类型可以是 float32 或 float64，其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值，其关系应符合 ``0 <= x1 < x2 && 0 <= y1 < y2``。
    - **iou_threshold** (float32，可选) - 用于判断两个框是否重叠的 IoU 门限值。如果 IoU(box1, box2) > threshold， box1 和 box2 将被认为是重叠框。默认为：0.3。
    - **scores** (Tensor，可选) - 与 boxes 参数对应的 score，它应当是一个形状为[num_boxes]的 1-D Tensor。数据类型可以是 float32 或 float64。默认为：None。
    - **category_idxs** (Tensor，可选) - 与 boxes 参数对应的类别编号，它应当是一个形状为[num_boxes]的 1-D Tensor。数据类型为 int64。默认为：None。
    - **categories** (List，可选) - 类别列表，它的每个元素应该是唯一的，满足 ``categories == paddle.unique(class_idxs)``。默认为：None。
    - **top_k** (int64，可选) - 需要返回的分数最高的 boxes 索引数量。该值须小于等于 num_boxes。默认为：None。


返回
:::::::::
    Tensor，被 NMS 保留的检测边界框的索引，它应当是一个形状为[num_boxes]的 1-D Tensor。


代码示例
:::::::::
COPY-FROM: paddle.vision.ops.nms
