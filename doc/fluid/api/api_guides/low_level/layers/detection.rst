..  _api_guide_detection:

图像检测
#########

PaddlePaddle Fluid在图像检测过程中实现了多个特有的操作。
这些操作仅在某些模型中存在，因此将此类api统一存在 detection 模块中。以下分模型介绍各个api：


SSD
----------------
* Prior Box：根据不同参数为每个输入位置生成一系列候选框。API Reference 请参考 :ref:`api_fluid_layers_prior_box`

* multi_box_head ：得到不同prior box的位置和置信度。API Reference 请参考 :ref:`api_fluid_layers_multi_box_head`

* detection_output：对prioir box解码，通过多分类NMS得到检测结果。API Reference 请参考 :ref:`api_fluid_layers_detection_output`

* ssd_loss：通过位置偏移预测值，置信度，检测框位置和真实框位置和标签计算损失。API Reference 请参考 :ref:`api_fluid_layers_ssd_loss`

* detection map: 利用mAP评估SSD网络模型。API Reference 请参考 :ref:`api_fluid_layers_detection_map`


Faster RCNN
-------------
* rpn_target_assign：通过anchor和真实框为anchor分配RPN网络的分类和回归目标值。API Reference 请参考 :ref:`api_fluid_layers_rpn_target_assign`

* anchor_generator：为每个位置生成一系列anchor。API Reference 请参考 :ref:`api_fluid_layers_anchor_generator`

* generate_proposal_labels: 通过generate_proposals得到的候选框和真实框得到RCNN部分的分类和回归的目标值。API Reference 请参考 :ref:`api_fluid_layers_generate_proposal_labels`

* generate_proposals: 对RPN网络输出box解码并筛选得到新的候选框。API Reference 请参考 :ref:`api_fluid_lauyers_generate_proposals`


OCR
---------
* roi_perspective_transform：对输入roi做透视变换。API Reference 请参考 :ref:`api_fluid_layers_roi_perspective_transform`

* polygon_box_transform：对不规则检测框进行坐标变换。API Reference 请参考 :ref:`api_fluid_layers_polygon_box_transform`

通用操作
-------------

图像检测中的一些通用操作，是对检测框的一系列操作，其中包括对检测框的编码解码（box_coder）；比较两个检测框并进行匹配（iou_similarity，bipartite_match）；根据检测框和标签得到分类和回归目标值（target_assign）：

* bipartite_match： 通过贪心二分匹配算法得到每一列中距离最大的一行。API Reference 请参考 :ref:`api_fluid_layers_bipartite_match`

* target_assign: 根据目标检测框和标签，分配分类和回归目标以及对应权重。API Reference 请参考 :ref:`api_fluid_layers_target_assign`

* iou_similarity：计算两组框的IOU值。API Reference 请参考 :ref:`api_fluid_layers_iou_similarity`

* box_coder：对检测框进行编码，解码。API Reference 请参考 :ref:`api_fluid_layers_box_coder`

