..  _api_guide_detection:


图像检测
#########

PaddlePaddle Fluid在图像检测任务中实现了多个特有的操作。以下分模型介绍各个api：

通用操作
-------------

图像检测中的一些通用操作，是对检测框的一系列操作，其中包括：

* 对检测框的编码，解码（box_coder）：实现候选框的位置值和对目标框位置偏移值的相互转化。API Reference 请参考 :ref:`api_fluid_layers_box_coder`

* 比较两个检测框并进行匹配：

  * iou_similarity：计算两组框的IOU值。API Reference 请参考 :ref:`api_fluid_layers_iou_similarity`

  * bipartite_match：通过贪心二分匹配算法得到每一列中距离最大的一行。API Reference 请参考 :ref:`api_fluid_layers_bipartite_match`

* 根据检测框和标签得到分类和回归目标值（target_assign）：通过匹配索引和非匹配索引得到目标值和对应权重。API Reference 请参考 :ref:`api_fluid_layers_target_assign`


Faster RCNN
-------------

`Faster RCNN <https://arxiv.org/abs/1506.01497>`_ 是典型的两阶段目标检测器，相较于传统提取区域的方法，Faster RCNN中RPN网络通过共享卷积层参数大幅提高提取区域的效率，并提出高质量的候选区域。RPN网络需要对输入anchor和真实值进行比较生成初选候选框，并对初选候选框分配分类和回归值，>需要如下四个特有api：

* rpn_target_assign：通过anchor和真实框为anchor分配RPN网络的分类和回归目标值。API Reference 请参考 :ref:`api_fluid_layers_rpn_target_assign`

* anchor_generator：为每个位置生成一系列anchor。API Reference 请参考 :ref:`api_fluid_layers_anchor_generator`

* generate_proposal_labels: 通过generate_proposals得到的候选框和真实框得到RCNN部分的分类和回归的目标值。API Reference 请参考 :ref:`api_fluid_layers_generate_proposal_labels`

* generate_proposals: 对RPN网络输出box解码并筛选得到新的候选框。API Reference 请参考 :ref:`api_fluid_layers_generate_proposals`


SSD
----------------

`SSD <https://arxiv.org/abs/1512.02325>`_ 全称Single Shot MultiBox Detector，是目标检测领域较新且效果较好的检测算法之一，具有检测速度快且检测精度高的特点。与两阶段的检测方法不同，单阶段目标检测并不进行区域推荐，而是直接从特征图回归出目标的边界框和分类概率。SSD网络对六个尺度特>征图计算损失，进行预测，需要如下五种特有api：

* Prior Box：根据不同参数为每个输入位置生成一系列候选框。API Reference 请参考 :ref:`api_fluid_layers_prior_box`

* multi_box_head ：得到不同prior box的位置和置信度。API Reference 请参考 :ref:`api_fluid_layers_multi_box_head`

* detection_output：对prioir box解码，通过多分类NMS得到检测结果。API Reference 请参考 :ref:`api_fluid_layers_detection_output`

* ssd_loss：通过位置偏移预测值，置信度，检测框位置和真实框位置和标签计算损失。API Reference 请参考 :ref:`api_fluid_layers_ssd_loss`

* detection map: 利用mAP评估SSD网络模型。API Reference 请参考 :ref:`api_fluid_layers_detection_map`

OCR
---------

场景文字识别是在图像背景复杂、分辨率低下、字体多样、分布随意等情况下，将图像信息转化为文字序列的过程，可认为是一种特别的翻译过程：将图像输入翻译为自然语言输出。OCR任务中需要对检测框进行不规则变换，其中需要如下两个api：

* roi_perspective_transform：对输入roi做透视变换。API Reference 请参考 :ref:`api_fluid_layers_roi_perspective_transform`

* polygon_box_transform：对不规则检测框进行坐标变换。API Reference 请参考 :ref:`api_fluid_layers_polygon_box_transform`


