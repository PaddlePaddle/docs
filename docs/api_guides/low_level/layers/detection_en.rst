
.. _api_guide_detection_en:


Image Detection
#################

PaddlePaddle Fluid implements several unique operators for image detection tasks. This article introduces related APIs grouped by diverse model types.

General operations
--------------------

Some common operations in image detection are a series of operations on the bounding boxes, including:

* Encoding and decoding of the bounding box : Conversion between encoding and decoding between the two kinds of boxes. For example, the training phase encodes the prior box and the ground-truth box to obtain the training target value. For API Reference, please refer to :ref:`api_fluid_layers_box_coder`

* Compare the two bounding boxes and match them:

  * iou_similarity: Calculate the IOU value of the two sets of boxes. For API Reference, please refer to :ref:`api_fluid_layers_iou_similarity`

  * bipartite_match: Get the row with the largest distance in each column by the greedy binary matching algorithm. For API Reference, please refer to :ref:`api_fluid_layers_bipartite_match`

* Get classification and regression target values ​​(target_assign) based on the bounding boxes and labels: Get the target values and corresponding weights by matched indices and negative indices. For API Reference, please refer to :ref:`api_fluid_layers_target_assign`


Faster RCNN
-------------

`Faster RCNN <https://arxiv.org/abs/1506.01497>`_ is a typical dual-stage target detector. Compared with the traditional extraction method, the RPN network in Faster RCNN greatly improves the extraction efficiency by sharing convolution layer parameters, and proposes high-quality region proposals. The RPN network needs to compare the input anchor with the ground-truth value to generate a primary candidate region, and assigns a classification and regression value to the primary candidate box. The following four unique apis are required:

* rpn_target_assign: Assign the classification and regression target values ​​of the RPN network to the anchor through the anchor and the ground-truth box. For API Reference, please refer to :ref:`api_fluid_layers_rpn_target_assign`

* anchor_generator: Generate a series of anchors for each location. For API Reference, please refer to :ref:`api_fluid_layers_anchor_generator`

* generate_proposal_labels: Get the classification and regression target values ​​of the RCNN part through the candidate box and the ground-truth box obtained by generate_proposals. For API Reference, please refer to :ref:`api_fluid_layers_generate_proposal_labels`

* generate_proposals: Decode the RPN network output box and selects a new region proposal. For API Reference, please refer to :ref:`api_fluid_layers_generate_proposals`


SSD
----------------

`SSD <https://arxiv.org/abs/1512.02325>`_ , the acronym for Single Shot MultiBox Detector, is one of the latest and better detection algorithms in the field of target detection. It has the characteristics of fast detection speed and high detection accuracy. Unlike the dual-stage detection method, the single-stage target detection does not perform regional proposals, but directly returns the target's bounding box and classification probability from the feature map. The SSD network calculates the loss through six metrics of features maps and performs prediction. SSD requires the following five unique apis:

* Prior Box: Generate a series of candidate boxes for each input position based on different parameters. For API Reference, please refer to :ref:`api_fluid_layers_prior_box`

* multi_box_head : Get the position and confidence of different prior boxes. For API Reference, please refer to :ref:`api_fluid_layers_multi_box_head`

* detection_output: Decode the prior box and obtains the detection result by multi-class NMS. For API Reference, please refer to :ref:`api_fluid_layers_detection_output`

* ssd_loss: Calculate the loss by prediction value of position offset, confidence, bounding box position and ground-truth box position and label. For API Reference, please refer to :ref:`api_fluid_layers_ssd_loss`

* detection map: Evaluate the SSD network model using mAP. For API Reference, please refer to :ref:`api_fluid_layers_detection_map`

OCR
---------

Scene text recognition is a process of converting image information into a sequence of characters in the case of complex image background, low resolution, diverse fonts, random distribution and so on. It can be considered as a special translation process: translation of image input into natural language output. The OCR task needs to perform irregular transformation on the bounding box, which requires the following two APIs:

* roi_perspective_transform: Make a perspective transformation on the input RoI. For API Reference, please refer to :ref:`api_fluid_layers_roi_perspective_transform`

* polygon_box_transform: Coordinate transformation of the irregular bounding box. For API Reference, please refer to :ref:`api_fluid_layers_polygon_box_transform`
