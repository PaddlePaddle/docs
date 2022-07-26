.. _cn_api_fluid_layers_yolo_box:

yolo_box
-------------------------------

.. py:function:: paddle.fluid.layers.yolo_box(x, img_size, anchors, class_num, conf_thresh, downsample_ratio, clip_bbox=True,name=None)





该运算符基于YOLOv3网络的输出结果，生成YOLO检测框。

连接 yolo_box 网络的输出形状应为[N，C，H，W]，其中 H 和 W 相同，用来指定网格大小。对每个网格点预测给定的数目的框，这个数目记为 S，由 anchor 的数量指定。在第二维（通道维度）中，C应该等于S *（5 + class_num），class_num是源数据集中对象类别数目（例如coco数据集中的80），此外第二个（通道）维度中还有4个框位置坐标x，y，w，h，以及anchor box的one-hot key的置信度得分。

假设4个位置坐标是 :math:`t_x` ，:math:`t_y` ，:math:`t_w` ， :math:`t_h`
，则框的预测算法为：

.. math::

    b_x &= \sigma(t_x) + c_x\\
    b_y &= \sigma(t_y) + c_y\\
    b_w &= p_w e^{t_w}\\
    b_h &= p_h e^{t_h}\\

在上面的等式中，:math:`c_x` ， :math:`c_x` 是当前网格的左上角顶点坐标。:math:`p_w` ， :math:`p_h`  由anchors指定。

每个anchor预测框的第五通道的逻辑回归值表示每个预测框的置信度得分，并且每个anchor预测框的最后class_num通道的逻辑回归值表示分类得分。应忽略置信度低于conf_thresh的框。另外，框最终得分是置信度得分和分类得分的乘积。


.. math::

    score_{pred} = score_{conf} * score_{class}


参数
::::::::::::

    - **x** （Variable） -  YoloBox的输入张量是一个4-D张量，形状为[N，C，H，W]。第二维（C）存储每个anchor box位置坐标，每个anchor box的置信度分数和one hot key。通常，X应该是YOLOv3网络的输出。数据类型为float32或float64
    - **img_size** （Variable） -  YoloBox的图像大小张量，这是一个形状为[N，2]的二维张量。该张量保持每个输入图像的高度和宽度，用于对输出图像按输入图像比例调整输出框的大小。数据类型为int32。
    - **anchors** （list | tuple） - anchor的宽度和高度，它将逐对解析
    - **class_num** （int） - 要预测的类数
    - **conf_thresh** （float） - 检测框的置信度得分阈值。置信度得分低于阈值的框应该被忽略
    - **downsample_ratio** （int） - 从网络输入到YoloBox操作输入的下采样率，因此应依次为第一个，第二个和第三个YoloBox运算设置该值为32,16,8
    - **clip_bbox** （bool） - 是否将输出的bbox裁剪到 :attr:`img_size` 范围内，默认为True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 
     1. 框的坐标，形为[N，M，4]的三维张量
     2. 框的分类得分，形为 [N，M，class_num]的三维张量

返回类型
::::::::::::
   变量（Variable）

抛出异常
::::::::::::

    - TypeError  -  yolov_box的输入x必须是Variable
    - TypeError  -  yolo框的anchors参数必须是list或tuple
    - TypeError  -  yolo box的class_num参数必须是整数
    - TypeError  -  yolo框的conf_thresh参数必须是一个浮点数

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.yolo_box