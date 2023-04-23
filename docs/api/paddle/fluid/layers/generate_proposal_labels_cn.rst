.. _cn_api_fluid_layers_generate_proposal_labels:

generate_proposal_labels
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposal_labels(rpn_rois, gt_classes, is_crowd, gt_boxes, im_info, batch_size_per_im=256, fg_fraction=0.25, fg_thresh=0.25, bg_thresh_hi=0.5, bg_thresh_lo=0.0, bbox_reg_weights=[0.1, 0.1, 0.2, 0.2], class_nums=None, use_random=True, is_cls_agnostic=False, is_cascade_rcnn=False)




**注意：该 OP 无对应的反向 OP**

该 OP 根据 RPN 预测产出的 bounding boxes 和 groundtruth，抽取出用来计算 loss 的 foreground boxes and background boxes。

RPN 的输出经过 ``generate_proposals OP`` 的处理，产出 ``RPN RoIs``，即该 OP 的输入。然后，在该 OP 中按以下规则对 ``RPN RoIs`` 进行分类：

- 与某个 groundtruth 的重叠面积大于 ``fg_thresh``，则该 box 被标记为 foreground box。
- 与某个 groundtruth 的重叠面积大于 ``bg_thresh_lo`` 且小于 ``bg_thresh_hi``，则该 box 被标记为 background box。

按上述规则筛选出一批 boxes 后，在对这些 boxes 做随机采样，以保证 foreground boxes 的数量不高于 batch_size_per_im * fg_fraction。

对最终得到的 boxes，我们给它们分配类别标签和回归目标(box label)，并产出 ``bboxInsideWeights`` 和 ``BboxOutsideWeights`` 。

参数
::::::::::::

  - **rpn_rois** (Variable) – Shape 为 ``[N, 4]`` 的 2-D LoDTensor。其中，N 为 RoIs 的个数。每个 RoI 以 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 的格式表示，其中，:math:`x_{min}` 和 :math:`y_{min}` 为 RoI 的左上角坐标，:math:`x_{max}` 和 :math:`y_{max}` 为 RoI 的右下角坐标。数据类型支持 float32 和 float64。
  - **gt_classes** (Variable) – Shape 为 ``[M, 1]`` 的 2-D LoDTensor，M 为 groundtruth boxes 的数量。用于表示 groundtruth boxes 的类别 ID。数据类型支持 int32。
  - **is_crowd** (Variable) –Shape 为 ``[M, 1]`` 的 2-D LoDTensor，M 为 groundtruth boxes 的数量。用于标记 boxes 是否是 crowd。数据类型支持 int32。
  - **gt_boxes** (Variable) – Shape 为 ``[M, 4]`` 的 2-D LoDTensor，M 为 groundtruth boxes 的数量。每个 box 以 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 的格式表示。
  - **im_info** (Variable) - Shape 为 ``[N，3]`` 的 2-DTensor，表示原始图像的大小信息。信息包含原始图像宽、高和 ``feature map`` 相对于原始图像缩放的比例。
  - **batch_size_per_im** (int，可选) – 整型数字。每张图片抽取出的的 RoIs 的数目。数据类型支持 int32。缺省值为 256。
  - **fg_fraction** (float，可选) – 浮点数值。在单张图片中，foreground boxes 占所有 boxes 的比例。数据类型支持 float32。缺省值为 0.25。
  - **fg_thresh** (float，可选) – 浮点数值。foreground 重叠阀值，用于筛选 foreground boxes。数据类型支持 float32。缺省值为 0.25。
  - **bg_thresh_hi** (float，可选) – 浮点数值。background 重叠阀值的上界，用于筛选 background boxes。数据类型支持 float32。缺省值为 0.5。
  - **bg_thresh_lo** (float，可选) – 浮点数值。background 重叠阀值的下界，用于筛选 background boxes。数据类型支持 float32。缺省值为 0.0。
  - **bbox_reg_weights** (list|tuple，可选) – 列表或元组。Box 回归权重。数据类型支持 float32。缺省值为[0.1,0.1,0.2,0.2]。
  - **class_nums** (int，可选) – 整型数字。类别数目。数据类型支持 int32。缺省值为 None。
  - **use_random** (bool，可选) – 布尔类型。是否使用随机采样来选择 foreground boxes 和 background boxes。缺省值为 True。
  - **is_cls_agnostic** (bool，可选)- 布尔类型。是否忽略类别，只做位置回归。缺省值为 False。
  - **is_cascade_rcnn** (bool，可选)- 布尔类型。是否为 cascade RCNN 模型，为 True 时采样策略发生变化。缺省值为 False。


返回
::::::::::::
元组，格式为 ``(rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights)``，其中，各个元素解释如下：

- **rois** - Shape 为 ``[P, 4]`` 的 2-D LoDTensor，P 一般是 ``batch_size_per_im * batch_size``。每个 RoIs 以 ``[xmin, ymin, xmax, ymax]`` 的格式表示。数据类型与 ``rpn_rois`` 一致。
- **labels_int32** - Shape 为 ``[P, 1]`` 的 2-D LoDTensor，P 一般是 ``batch_size_per_im * batch_size``。表示每个 RoI 的类别 ID。数据类型为 int32。
- **bbox_targets** - Shape 为 ``[P, 4 * class_num]`` 的 2-D LoDTensor，表示所有 RoIs 的回归目标（box label）。数据类型与 ``rpn_rois`` 一致。
- **bbox_inside_weights** - Shape 为 ``[P, 4 * class_num]`` 的 2-D LoDTensor。foreground boxes 回归 loss 的权重。数据类型与 ``rpn_rois`` 一致。
- **bbox_outside_weights** - Shape 为 ``[P, 4 * class_num]`` 的 2-D LoDTensor。回归 loss 的权重。数据类型与 ``rpn_rois`` 一致。

返回类型
::::::::::::
元组



代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.generate_proposal_labels
