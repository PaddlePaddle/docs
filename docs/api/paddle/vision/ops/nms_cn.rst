.. _cn_api_paddle_vision_ops_nms:

nms
-------------------------------

.. py:function:: paddle.vision.ops.nms(boxes, iou_threshold=0.3, scores=None, category_idxs=None, categories=None,top_k=None)

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
    - scores(Tensor) - 与boxes参数对应的score，它应当是一个形状为(num_boxes)的1-D Tensor。数据类型可以是float32或float64。
    - category_idxs(Tensor) - 与boxes参数对应的类别编号，它应当是一个形状为(num_boxes)的1-D Tensor。数据类型为int64。
    - categories(List) - 类别列表，它的每个元素应该是唯一的，满足categories == paddle.unique(class_idxs).
    - top_k(int64) - 需要返回的分数最高的boxes索引数量。该值须小于等于num_boxes。


返回
:::::::::
    - Tensor - 被NMS保留的检测边界框的索引，它应当是一个形状为(num_boxes)的1-D Tensor。


代码示例
:::::::::

..  code-block:: python

    import paddle
    import numpy as np

    boxes = np.random.rand(4, 4).astype('float32')
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    # [[0.46517828 0.90615124 1.3940141  1.8840896 ]
    # [0.74385834 0.8236293  1.4048514  1.3868837 ]
    # [0.39436954 0.18261194 1.3834884  0.38191944]
    # [0.9617653  0.40089446 1.2982695  1.398673  ]]

    out =  paddle.vision.ops.nms(paddle.to_tensor(boxes), 0.1)
    # [0, 2, 0, 0])

..  code-block:: python

    import paddle
    import numpy as np

    boxes = np.random.rand(4, 4).astype('float32')
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    # [[0.46517828 0.90615124 1.3940141  1.8840896 ]
    # [0.74385834 0.8236293  1.4048514  1.3868837 ]
    # [0.39436954 0.18261194 1.3834884  0.38191944]
    # [0.9617653  0.40089446 1.2982695  1.398673  ]]

    scores = np.random.rand(4).astype('float32')
    # [0.20447887, 0.6679728 , 0.00704206, 0.14359951]
    categories = [0, 1, 2, 3]
    category_idxs = np.random.choice(categories, 4)                        
    # [1, 3, 0, 1]

    out =  paddle.vision.ops.batched_nms(paddle.to_tensor(boxes), 
                                            paddle.to_tensor(scores), 
                                            paddle.to_tensor(category_idxs), 
                                            categories, 
                                            0.1, 
                                            4)
    # [1, 0, 2]