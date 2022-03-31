.. _cn_api_paddle_vision_ops_nms:

nms
-------------------------------

.. py:function:: paddle.vision.ops.nms(boxes, iou_threshold=0.3, scores=None, category_idxs=None, categories=None, top_k=None)

非极大抑制(non-maximum suppression, NMS)用于在目标检测应用对检测边界框(bounding box)中搜索局部最大值，即只保留处于同一检测目标位置处重叠的框中分数最大的一个框。IoU(Intersection Over Union) 被用于判断两个框是否重叠，该值大于门限值则被认为两个框重叠。其计算公式如下：

.. math:: 

    IoU = \frac{intersection\_area(box1, box2)}{union\_area(box1, box2)}

如果参数scores不为None，输入的boxes会首先按照它们对应的score降序排序，否则将默认输入的boxes为排好序的。
如果category_idxs和categories不为None，分类NMS将会被执行，也就是说，nms过程会在每一个类别的框当中分别进行计算，计算结果会被组合起来然后按照得分倒序排列。
如果top_k不为None的话，排序的计算结果中仅有前k个元素会被返回，否则会返回所有的元素。

参数
:::::::::
    - boxes(Tensor) - 待进行计算的框坐标，它应当是一个形状为(num_boxes, 4)的2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。数据类型可以是float32或float64.请注意，nms无需传入score信息，因此请提前按照框的score对boxes进行逆序排列。
    - iou_threshold(float32) - 用于判断两个框是否重叠的IoU门限值。 如果IoU(box1, box2) > threshold， box1和box2将被认为是重叠框。
    - scores(Tensor，可选) - 与boxes参数对应的score，它应当是一个形状为(num_boxes)的1-D Tensor。数据类型可以是float32或float64。
    - category_idxs(Tensor，可选) - 与boxes参数对应的类别编号，它应当是一个形状为(num_boxes)的1-D Tensor。数据类型为int64。
    - categories(List，可选) - 类别列表，它的每个元素应该是唯一的，满足categories == paddle.unique(class_idxs).
    - top_k(int64，可选) - 需要返回的分数最高的boxes索引数量。该值须小于等于num_boxes。


返回
:::::::::
    - Tensor - 被NMS保留的检测边界框的索引，它应当是一个形状为(num_boxes)的1-D Tensor。


代码示例
:::::::::
COPY-FROM: paddle.vision.ops.nms