.. _cn_api_paddle_vision_ops_batched_nms:

batched_nms
-------------------------------

.. py:function:: paddle.vision.ops.batched_nms(boxes, scores, category_idxs, categories, iou_threshold, top_k)

在batched_nms中，nms过程会在每一个类别的框当中分别进行计算，计算结果会被组合起来然后按照得分倒序排列。了解nms可参考 :ref:`cn_api_paddle_vision_ops_nms`

参数
:::::::::
    - boxes(Tensor) - 待进行计算的框坐标，它应当是一个形状为(num_boxes, 4)的2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。数据类型可以是float32或float64。
    - scores(Tensor) - 与boxes参数对应的score，它应当是一个形状为(num_boxes)的1-D Tensor。数据类型可以是float32或float64。
    - category_idxs(Tensor) - 与boxes参数对应的类别编号，它应当是一个形状为(num_boxes)的1-D Tensor。数据类型为int64。
    - categories(List) - 类别列表，它的每个元素应该是唯一的，满足categories == paddle.unique(class_idxs).
    - iou_threshold(float32) - 用于判断两个框是否重叠的IoU门限值。 如果IoU(box1, box2) > threshold， box1和box2将被认为是重叠框。
    - top_k(int64) - 需要返回的分数最高的boxes索引数量。该值须小于等于num_boxes。

返回
:::::::::
    - Tensor - 被NMS保留的框的索引，按照框对应score倒序排列。

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