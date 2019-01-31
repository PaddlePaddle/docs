
.. _api_guide_detection_en:


Image detection
#################

PaddlePaddle Fluid implements several unique operations in image inspection tasks. The following sub-models introduce each api:

General operation
--------------------

Some common operations in image detection are a series of operations on the detection frame, including:

* Encoding and decoding of the detection box (box_coder): Conversion between encoding and decoding between the two frames. For example, the training phase encodes the a priori box and the real box to obtain the training target value. About API Reference, please refer to :ref:`api_fluid_layers_box_coder`

* Compare the two detection boxes and match them:

  * iou_similarity: Calculates the IOU value of the two sets of boxes. About API Reference, please refer to :ref:`api_fluid_layers_iou_similarity`

  * bipartite_match: Get the row with the largest distance in each column by the greedy binary matching algorithm. About API Reference, please refer to :ref:`api_fluid_layers_bipartite_match`

* Get classification and regression target values ​​(target_assign) according to the detection frame and label: get the target value and corresponding weight by matching index and non-matching index. About API Reference please refer to :ref:`api_fluid_layers_target_assign`


Faster RCNN
-------------

`Faster RCNN <https://arxiv.org/abs/1506.01497>`_ is a typical two-stage target detector. Compared to the traditional extraction region method, the RPN network in Faster RCNN greatly improves the extraction by sharing convolutional layer parameters. Regional efficiency and propose high quality candidate areas. The RPN network needs to compare the input anchor with the real value to generate a primary candidate box, and assign a classification and regression value to the primary candidate box. The following four unique apis are required:

* rpn_target_assign: Assigns the classification and regression target values ​​of the RPN network to the anchor through the anchor and the real box. About API Reference, please refer to :ref:`api_fluid_layers_rpn_target_assign`

* anchor_generator: Generates a series of anchors for each location. About API Reference, please refer to :ref:`api_fluid_layers_anchor_generator`

* generate_proposal_labels: The candidate box and the real box obtained by generate_proposals get the classification and regression target values ​​of the RCNN part. About API Reference, please refer to :ref:`api_fluid_layers_generate_proposal_labels`

* generate_proposals: Decodes the RPN network output box and filters it to get a new candidate box. About API Reference, please refer to :ref:`api_fluid_layers_generate_proposals`


SSD
----------------

`SSD <https://arxiv.org/abs/1512.02325>`_ Full name Single Shot MultiBox Detector is one of the newer and better detection algorithms in the field of target detection. It has the characteristics of fast detection speed and high detection precision. Unlike the two-stage detection method, the single-stage target detection does not perform regional recommendation, but directly returns the target's bounding box and classification probability from the feature map. The SSD network calculates the loss for six scale features and maps, and requires the following five unique apis:

* Prior Box: Generates a series of candidate boxes for each input position based on different parameters. About API Reference, please refer to :ref:`api_fluid_layers_prior_box`

* multi_box_head : Get the position and confidence of different prior boxes. About API Reference, please refer to :ref:`api_fluid_layers_multi_box_head`

* detection_output: Decodes the prioir box and obtains the detection result by multi-class NMS. About API Reference, please refer to :ref:`api_fluid_layers_detection_output`

* ssd_loss: Calculate the loss by position offset prediction, confidence, detection frame position and real frame position and label. About API Reference, please refer to :ref:`api_fluid_layers_ssd_loss`

* detection map: Evaluate the SSD network model using mAP. About API Reference, please refer to :ref:`api_fluid_layers_detection_map`

OCR
---------

Scene text recognition is a process of converting image information into a sequence of characters in the case of complex image background, low resolution, diverse fonts, random distribution, etc. It can be considered as a special translation process: translation of image input into natural language output. The OCR task needs to perform irregular transformation on the detection frame, which requires the following two apis:

* roi_perspective_transform: Make a perspective transformation on the input roi. About API Reference, please refer to :ref:`api_fluid_layers_roi_perspective_transform`

* polygon_box_transform: Coordinate transformation of the irregular detection frame. About API Reference, please refer to :ref:`api_fluid_layers_polygon_box_transform`