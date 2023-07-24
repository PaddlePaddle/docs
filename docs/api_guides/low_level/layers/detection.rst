..  _api_guide_detection:


图像检测
#########

PaddlePaddle Fluid 在图像检测任务中实现了多个特有的操作。以下分模型介绍各个 api：

通用操作
-------------

图像检测中的一些通用操作，是对检测框的一系列操作，其中包括：

* 对检测框的编码，解码（box_coder）：实现两种框之间编码和解码的转换。例如训练阶段对先验框和真实框进行编码得到训练目标值。API Reference 请参考 :ref:`cn_api_fluid_layers_box_coder`

* 比较两个检测框并进行匹配：

  * iou_similarity：计算两组框的 IOU 值。API Reference 请参考 :ref:`cn_api_fluid_layers_iou_similarity`

  * bipartite_match：通过贪心二分匹配算法得到每一列中距离最大的一行。API Reference 请参考 :ref:`cn_api_fluid_layers_bipartite_match`

* 根据检测框和标签得到分类和回归目标值（target_assign）：通过匹配索引和非匹配索引得到目标值和对应权重。API Reference 请参考 :ref:`cn_api_fluid_layers_target_assign`

* 对检测框进行后处理：

  * box_clip: 将检测框剪切到指定大小。API Reference 请参考 :ref:`cn_api_fluid_layers_box_clip`

  * multiclass_nms: 对边界框和评分进行多类非极大值抑制。API Reference 请参考 :ref:`cn_api_fluid_layers_multiclass_nms`


RCNN
-------------

RCNN 系列模型是两阶段目标检测器，其中包含`Faster RCNN <https://arxiv.org/abs/1506.01497>`_，`Mask RCNN <https://arxiv.org/abs/1703.06870>`_，相较于传统提取区域的方法，RCNN 中 RPN 网络通过共享卷积层参数大幅提高提取区域的效率，并提出高质量的候选区域。RPN 网络需要对输入 anchor 和真实值进行比较生成初选候选框，并对初选候选框分配分类和回归值，需要如下五个特有 api：

* rpn_target_assign：通过 anchor 和真实框为 anchor 分配 RPN 网络的分类和回归目标值。API Reference 请参考 :ref:`cn_api_fluid_layers_rpn_target_assign`

* anchor_generator：为每个位置生成一系列 anchor。API Reference 请参考 :ref:`cn_api_fluid_layers_anchor_generator`

* generate_proposal_labels: 通过 generate_proposals 得到的候选框和真实框得到 RCNN 部分的分类和回归的目标值。API Reference 请参考 :ref:`cn_api_fluid_layers_generate_proposal_labels`

* generate_proposals: 对 RPN 网络输出 box 解码并筛选得到新的候选框。API Reference 请参考 :ref:`cn_api_fluid_layers_generate_proposals`

* generate_mask_labels: 通过 generate_proposal_labels 得到的 RoI，和真实框对比后进一步筛选 RoI 并得到 Mask 分支的目标值。API Reference 请参考 :ref:`cn_api_fluid_layers_generate_mask_labels`

FPN
-------------

`FPN <https://arxiv.org/abs/1612.03144>`_ 全称 Feature Pyramid Networks, 采用特征金字塔做目标检测。 顶层特征通过上采样和低层特征做融合，并将 FPN 放在 RPN 网络中用于生成候选框，有效的提高检测精度，需要如下两种特有 api：

* collect_fpn_proposals: 拼接多层 RoI，同时选择分数较高的 RoI。API Reference 请参考 :ref:`cn_api_fluid_layers_collect_fpn_proposals`

* distribute_fpn_proposals: 将多个 RoI 依据面积分配到 FPN 的多个层级中。API Reference 请参考 :ref:`cn_api_fluid_layers_distribute_fpn_proposals`

SSD
----------------

`SSD <https://arxiv.org/abs/1512.02325>`_ 全称 Single Shot MultiBox Detector，是目标检测领域较新且效果较好的检测算法之一，具有检测速度快且检测精度高的特点。与两阶段的检测方法不同，单阶段目标检测并不进行区域推荐，而是直接从特征图回归出目标的边界框和分类概率。SSD 网络对六个尺度特>征图计算损失，进行预测，需要如下五种特有 api：

* 根据不同参数为每个输入位置生成一系列候选框。

  * prior box: API Reference 请参考 :ref:`cn_api_fluid_layers_prior_box`

  * density_prior box: API Reference 请参考 :ref:`cn_api_fluid_layers_density_prior_box`

* multi_box_head ：得到不同 prior box 的位置和置信度。API Reference 请参考 :ref:`cn_api_fluid_layers_multi_box_head`

* detection_output：对 prior box 解码，通过多分类 NMS 得到检测结果。API Reference 请参考 :ref:`cn_api_fluid_layers_detection_output`

* ssd_loss：通过位置偏移预测值，置信度，检测框位置和真实框位置和标签计算损失。API Reference 请参考 :ref:`cn_api_fluid_layers_ssd_loss`

* detection_map: 利用 mAP 评估 SSD 网络模型。API Reference 请参考 :ref:`cn_api_fluid_layers_detection_map`

YOLO V3
---------------

`YOLO V3 <https://arxiv.org/abs/1804.02767>`_ 是单阶段目标检测器，同时具备了精度高，速度快的特点。对特征图划分多个区块，每个区块得到坐标位置和置信度。采用了多尺度融合的方式预测以得到更高的训练精度，需要如下两种特有 api：

* yolo_box: 从 YOLOv3 网络的输出生成 YOLO 检测框。API Reference 请参考 :ref:`cn_api_fluid_layers_yolo_box`

* yolov3_loss：通过给定的预测结果和真实框生成 yolov3 损失。API Reference 请参考 :ref:`cn_api_fluid_layers_yolov3_loss`

RetinaNet
---------------

`RetinaNet <https://arxiv.org/abs/1708.02002>`_ 是单阶段目标检测器，引入 Focal Loss 和 FPN 后，能以更快的速率实现与双阶段目标检测网络近似或更优的效果，需要如下三种特有 api：

* sigmoid_focal_loss: 用于处理单阶段检测器中类别不平均问题的损失。API Reference 请参考 :ref:`cn_api_fluid_layers_sigmoid_focal_loss`

* retinanet_target_assign: 对给定 anchor 和真实框，为每个 anchor 分配分类和回归的目标值，用于训练 RetinaNet。API Reference 请参考 :ref:`cn_api_fluid_layers_retinanet_target_assign`

* retinanet_detection_output: 对检测框进行解码，并做非极大值抑制后得到检测输出。API Reference 请参考 :ref:`cn_api_fluid_layers_retinanet_detection_output`

OCR
---------

场景文字识别是在图像背景复杂、分辨率低下、字体多样、分布随意等情况下，将图像信息转化为文字序列的过程，可认为是一种特别的翻译过程：将图像输入翻译为自然语言输出。OCR 任务中需要对检测框进行不规则变换，其中需要如下两个 api：

* roi_perspective_transform：对输入 roi 做透视变换。API Reference 请参考 :ref:`cn_api_fluid_layers_roi_perspective_transform`

* polygon_box_transform：对不规则检测框进行坐标变换。API Reference 请参考 :ref:`cn_api_fluid_layers_polygon_box_transform`
