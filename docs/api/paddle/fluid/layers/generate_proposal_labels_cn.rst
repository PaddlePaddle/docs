.. _cn_api_fluid_layers_generate_proposal_labels:

generate_proposal_labels
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposal_labels(rpn_rois, gt_classes, is_crowd, gt_boxes, im_info, batch_size_per_im=256, fg_fraction=0.25, fg_thresh=0.25, bg_thresh_hi=0.5, bg_thresh_lo=0.0, bbox_reg_weights=[0.1, 0.1, 0.2, 0.2], class_nums=None, use_random=True, is_cls_agnostic=False, is_cascade_rcnn=False)




**注意：该OP无对应的反向OP**

该OP根据RPN预测产出的bounding boxes和groundtruth，抽取出用来计算loss的foreground boxes and background boxes。

RPN的输出经过 ``generate_proposals OP`` 的处理，产出 ``RPN RoIs``，即该OP的输入。然后，在该OP中按以下规则对 ``RPN RoIs`` 进行分类：

- 与某个groundtruth的重叠面积大于 ``fg_thresh``，则该box被标记为foreground box。
- 与某个groundtruth的重叠面积大于 ``bg_thresh_lo`` 且小于 ``bg_thresh_hi``，则该box被标记为background box。

按上述规则筛选出一批boxes后，在对这些boxes做随机采样，以保证foreground boxes的数量不高于batch_size_per_im * fg_fraction。

对最终得到的boxes，我们给它们分配类别标签和回归目标(box label)，并产出 ``bboxInsideWeights`` 和 ``BboxOutsideWeights`` 。

参数
::::::::::::

  - **rpn_rois** (Variable) – Shape为 ``[N, 4]`` 的2-D LoDTensor。其中，N为RoIs的个数。每个RoI以 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 的格式表示，其中，:math:`x_{min}` 和 :math:`y_{min}` 为RoI的左上角坐标，:math:`x_{max}` 和 :math:`y_{max}` 为RoI的右下角坐标。数据类型支持float32和float64。
  - **gt_classes** (Variable) – Shape为 ``[M, 1]`` 的2-D LoDTensor，M为groundtruth boxes的数量。用于表示groundtruth boxes的类别ID。数据类型支持int32。
  - **is_crowd** (Variable) –Shape为 ``[M, 1]`` 的2-D LoDTensor，M为groundtruth boxes的数量。用于标记boxes是否是crowd。数据类型支持int32。
  - **gt_boxes** (Variable) – Shape为 ``[M, 4]`` 的2-D LoDTensor，M为groundtruth boxes的数量。每个box以 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 的格式表示。
  - **im_info** (Variable) - Shape为 ``[N，3]`` 的2-D张量，表示原始图像的大小信息。信息包含原始图像宽、高和 ``feature map`` 相对于原始图像缩放的比例。
  - **batch_size_per_im** (int，可选) – 整型数字。每张图片抽取出的的RoIs的数目。数据类型支持int32。缺省值为256。
  - **fg_fraction** (float，可选) – 浮点数值。在单张图片中，foreground boxes占所有boxes的比例。数据类型支持float32。缺省值为0.25。
  - **fg_thresh** (float，可选) – 浮点数值。foreground重叠阀值，用于筛选foreground boxes。数据类型支持float32。缺省值为0.25。
  - **bg_thresh_hi** (float，可选) – 浮点数值。background重叠阀值的上界，用于筛选background boxes。数据类型支持float32。缺省值为0.5。
  - **bg_thresh_lo** (float，可选) – 浮点数值。background重叠阀值的下界，用于筛选background boxes。数据类型支持float32。缺省值为0.0。
  - **bbox_reg_weights** (list|tuple，可选) – 列表或元组。Box 回归权重。数据类型支持float32。缺省值为[0.1,0.1,0.2,0.2]。
  - **class_nums** (int，可选) – 整型数字。类别数目。数据类型支持int32。缺省值为None。
  - **use_random** (bool，可选) – 布尔类型。是否使用随机采样来选择foreground boxes和background boxes。缺省值为True。
  - **is_cls_agnostic** (bool，可选)- 布尔类型。是否忽略类别，只做位置回归。缺省值为False。
  - **is_cascade_rcnn** (bool，可选)- 布尔类型。是否为 cascade RCNN 模型，为True时采样策略发生变化。缺省值为False。


返回
::::::::::::
元组，格式为 ``(rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights)``，其中，各个元素解释如下：

- **rois** - Shape为 ``[P, 4]`` 的2-D LoDTensor，P一般是 ``batch_size_per_im * batch_size``。每个RoIs以 ``[xmin, ymin, xmax, ymax]`` 的格式表示。数据类型与 ``rpn_rois`` 一致。
- **labels_int32** - Shape为 ``[P, 1]`` 的2-D LoDTensor，P一般是 ``batch_size_per_im * batch_size``。表示每个RoI的类别ID。数据类型为int32。
- **bbox_targets** - Shape为 ``[P, 4 * class_num]`` 的2-D LoDTensor，表示所有RoIs的回归目标（box label）。数据类型与 ``rpn_rois`` 一致。
- **bbox_inside_weights** - Shape为 ``[P, 4 * class_num]`` 的2-D LoDTensor。foreground boxes回归loss的权重。数据类型与 ``rpn_rois`` 一致。
- **bbox_outside_weights** - Shape为 ``[P, 4 * class_num]`` 的2-D LoDTensor。回归loss的权重。数据类型与 ``rpn_rois`` 一致。

返回类型
::::::::::::
元组



代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.generate_proposal_labels