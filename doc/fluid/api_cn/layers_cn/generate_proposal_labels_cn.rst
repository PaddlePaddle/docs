.. _cn_api_fluid_layers_generate_proposal_labels:

generate_proposal_labels
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposal_labels(rpn_rois, gt_classes, is_crowd, gt_boxes, im_info, batch_size_per_im=256, fg_fraction=0.25, fg_thresh=0.25, bg_thresh_hi=0.5, bg_thresh_lo=0.0, bbox_reg_weights=[0.1, 0.1, 0.2, 0.2], class_nums=None, use_random=True, is_cls_agnostic=False, is_cascade_rcnn=False)

**该函数可以应用于 Faster-RCNN 网络，生成建议标签。**

该函数可以根据 ``GenerateProposals`` 的输出结果，即bounding boxes（区域框），groundtruth（正确标记数据）来对foreground boxes和background boxes进行采样，并计算loss值。

RpnRois 是RPN的输出box， 并由 ``GenerateProposals`` 来进一步处理, 这些box将与groundtruth boxes合并， 并根据 ``batch_size_per_im`` 和 ``fg_fraction`` 进行采样。

如果一个实例具有大于 ``fg_thresh`` (前景重叠阀值)的正确标记重叠，那么它会被认定为一个前景样本。
如果一个实例具有的正确标记重叠大于 ``bg_thresh_lo`` 且小于 ``bg_thresh_hi`` (详见参数说明)，那么它将被认定为一个背景样本。
在所有前景、背景框（即Rois regions of interest 直译：有意义的区域）被选择后，我们接着采用随机采样的方法来确保前景框数量不多于 batch_size_per_im * fg_fraction 。

对Rois中的每个box, 我们给它分配类标签和回归目标(box label)。最后 ``bboxInsideWeights`` 和 ``BboxOutsideWeights`` 用来指明是否它将影响训练loss值。

参数:
  - **rpn_rois** (Variable) – 形为[N, 4]的二维LoDTensor。 N 为 ``GenerateProposals`` 的输出结果, 其中各元素为 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 格式的边界框
  - **gt_classes** (Variable) – 形为[M, 1]的二维LoDTensor。 M 为正确标记数据数目, 其中各元素为正确标记数据的类别标签
  - **is_crowd** (Variable) – 形为[M, 1]的二维LoDTensor。M 为正确标记数据数目, 其中各元素为一个标志位，表明一个正确标记数据是不是crowd
  - **gt_boxes** (Variable) – 形为[M, 4]的二维LoDTensor。M 为正确标记数据数目, 其中各元素为 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 格式的边界框
  - **im_info** (Variable) – 形为[B, 3]的二维LoDTensor。B 为输入图片的数目, 各元素由 im_height, im_width, im_scale 组成.
  - **batch_size_per_im** (int) – 每张图片的Rois batch数目
  - **fg_fraction** (float) – Foreground前景在 ``batch_size_per_im`` 中所占比例
  - **fg_thresh** (float) – 前景重叠阀值，用于选择foreground前景样本
  - **bg_thresh_hi** (float) – 背景重叠阀值的上界，用于筛选背景样本
  - **bg_thresh_lo** (float) – 背景重叠阀值的下界，用于筛选背景样本O
  - **bbox_reg_weights** (list|tuple) – Box 回归权重
  - **class_nums** (int) – 种类数目
  - **use_random** (bool) – 是否使用随机采样来选择foreground（前景）和background（背景） boxes（框）
  - **is_cls_agnostic** （bool）- 未知类别的bounding box回归，仅标识前景和背景框
  - **is_cascade_rcnn** （bool）- 是否为 cascade RCNN 模型，为True时采样策略发生变化

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    rpn_rois = fluid.layers.data(name='rpn_rois', shape=[2, 4],
                   append_batch_size=False, dtype='float32')
    gt_classes = fluid.layers.data(name='gt_classes', shape=[8, 1],
                   append_batch_size=False, dtype='float32')
    is_crowd = fluid.layers.data(name='is_crowd', shape=[8, 1],
                   append_batch_size=False, dtype='float32')
    gt_boxes = fluid.layers.data(name='gt_boxes', shape=[8, 4],
                   append_batch_size=False, dtype='float32')
    im_info = fluid.layers.data(name='im_info', shape=[10, 3],
                   append_batch_size=False, dtype='float32')
    rois, labels_int32, bbox_targets, bbox_inside_weights,
    bbox_outside_weights = fluid.layers.generate_proposal_labels(
                   rpn_rois, gt_classes, is_crowd, gt_boxes, im_info,
                   class_nums=10)











