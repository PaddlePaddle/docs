.. _cn_api_paddle_vision_ops_yolo_box:

yolo_box
-------------------------------

.. py:function:: paddle.vision.ops.yolo_box(x, img_size, anchors, class_num, conf_thresh, downsample_ratio, clip_bbox=True, name=None, scale_x_y=1.0, iou_aware=False, iou_aware_factor=0.5)

该运算符基于 YOLOv3 网络的输出结果，生成 YOLO 检测框。

连接 yolo_box 网络的输出形状应为[N，C，H，W]，其中 H 和 W 相同，用来指定网格大小。对每个网格点预测给定的数目的框，这个数目记为 S，由 anchor 的数量指定。在第二维（通道维度）中，C 应该等于 S *（5 + class_num），class_num 是源数据集中对象类别数目（例如 coco 数据集中的 80），此外第二个（通道）维度中还有 4 个框位置坐标 x，y，w，h，以及 anchor box 的 one-hot key 的置信度得分。

假设 4 个位置坐标是 :math:`t_x` ，:math:`t_y` ，:math:`t_w` ， :math:`t_h`，则框的预测算法为：

.. math::

    b_x &= \sigma(t_x) + c_x\\
    b_y &= \sigma(t_y) + c_y\\
    b_w &= p_w e^{t_w}\\
    b_h &= p_h e^{t_h}\\

在上面的等式中，:math:`c_x` ， :math:`c_y` 是当前网格的左上角顶点坐标。:math:`p_w` ， :math:`p_h`  由 anchors 指定。

每个 anchor 预测框的第五通道的逻辑回归值表示每个预测框的置信度得分，并且每个 anchor 预测框的最后 class_num 通道的逻辑回归值表示分类得分。应忽略置信度低于 conf_thresh 的框。另外，框最终得分是置信度得分和分类得分的乘积。


.. math::

    score_{pred} = score_{conf} * score_{class}

参数
:::::::::

    - **x** (Tensor) - YoloBox 的输入 Tensor 是一个 4-D Tensor，形状为[N，C，H，W]。第二维（C）存储每个 anchor box 位置坐标，每个 anchor box 的置信度分数和 one hot key。通常，X 应该是 YOLOv3 网络的输出。数据类型为 float32 或 float64。
    - **img_size** (Tensor) - YoloBox 的图像大小 Tensor，这是一个形状为[N，2]的二维 Tensor。该 Tensor 保持每个输入图像的高度和宽度，用于对输出图像按输入图像比例调整输出框的大小。数据类型为 int32。
    - **anchors** (list | tuple) - anchor 的宽度和高度，它将逐对解析。
    - **class_num** (int) - 要预测的类数。
    - **conf_thresh** (float) - 检测框的置信度得分阈值。置信度得分低于阈值的框应该被忽略。
    - **downsample_ratio** (int) - 从网络输入到 YoloBox 操作输入的下采样率，因此应依次为第一个，第二个和第三个 YoloBox 运算设置该值为 32,16,8
    - **clip_bbox** (bool，可选) - 是否将输出的 bbox 裁剪到 :attr:`img_size` 范围内，默认为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **scale_x_y** (float，可选) - 放缩解码边界框的中心点，默认值：1.0。
    - **iou_aware** (bool，可选) - 使用 IoU 置信度，默认值：False。
    - **iou_aware_factor** (bool，可选) - IoU 置信度因子，默认值：0.5。

返回
:::::::::

     1. 框的坐标，形为[N，M，4]的三维 Tensor；
     2. 框的分类得分，形为 [N，M，class_num]的三维 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.vision.ops.yolo_box
