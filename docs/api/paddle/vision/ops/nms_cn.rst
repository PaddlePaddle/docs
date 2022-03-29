.. _cn_api_paddle_vision_ops_nms:

nms
-------------------------------

.. py:function:: paddle.vision.ops.nms(boxes, threshold)

非极大抑制(non-maximum suppression, NMS)用于在目标检测应用中搜索局部最大值，即只保留处于同一检测目标位置处重叠的框中分数最大的一个框。IoU(Intersection Over Union) 被用于判断两个框是否重叠，该值大于门限值则被认为两个框重叠。其计算公式如下：

.. math:: 

    IoU = \frac{intersection\_area(box1, box2)}{union\_area(box1, box2)}

参数
:::::::::
    - boxes(Tensor) - 待进行计算的框坐标，它应当是一个形状为(num_boxes, 4)的2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出。其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值。数据类型可以是float32或float64.请注意，nms无需传入score信息，因此请提前按照框的score对boxes进行逆序排列。
    - threshold(float32) - 用于判断两个框是否重叠的IoU门限值。 如果IoU(box1, box2) > threshold， box1和box2将被认为是重叠框。


返回
:::::::::
    - Tensor - 被NMS保留的框的索引，它应当是一个形状为(num_boxes)的1-D Tensor。


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