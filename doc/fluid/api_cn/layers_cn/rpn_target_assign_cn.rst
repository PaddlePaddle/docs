.. _cn_api_fluid_layers_rpn_target_assign:

rpn_target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.rpn_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info, rpn_batch_size_per_im=256, rpn_straddle_thresh=0.0, rpn_fg_fraction=0.5, rpn_positive_overlap=0.7, rpn_negative_overlap=0.3, use_random=True)

在Faster-RCNN检测中为区域检测网络（RPN）分配目标层。

对于给定anchors和真实框之间的IoU重叠，该层可以为每个anchors做分类和回归，这些target labels用于训练RPN。classification targets是二进制的类标签（是或不是对象）。根据Faster-RCNN的论文，positive labels有两种anchors：

(i) anchor/anchors与真实框具有最高IoU重叠；

(ii) 具有IoU重叠的anchors高于带有任何真实框（ground-truth box）的rpn_positive_overlap0（0.7）。

请注意，单个真实框（ground-truth box）可以为多个anchors分配正标签。对于所有真实框（ground-truth box），非正向anchor是指其IoU比率低于rpn_negative_overlap（0.3）。既不是正也不是负的anchors对训练目标没有价值。回归目标是与positive anchors相关联而编码的图片真实框。

参数：
        - **bbox_pred** （Variable）- 是一个shape为[N，M，4]的3-D Tensor，表示M个边界框的预测位置。N是批量大小，每个边界框有四个坐标值，即[xmin，ymin，xmax，ymax]。
        - **cls_logits** （Variable）- 是一个shape为[N，M，1]的3-D Tensor，表示预测的置信度。N是批量大小，1是frontground和background的sigmoid，M是边界框的数量。
        - **anchor_box** （Variable）- 是一个shape为[M，4]的2-D Tensor，它拥有M个框，每个框可表示为[xmin，ymin，xmax，ymax]，[xmin，ymin]是anchor框的左上部坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是anchor框的右下部坐标。
        - **anchor_var** （Variable）- 是一个shape为[M，4]的2-D Tensor，它拥有anchor的expand方差。
        - **gt_boxes** （Variable）- 真实边界框是一个shape为[Ng，4]的2D LoDTensor，Ng是小批量输入的真实框（bbox）总数。
        - **is_crowd** （Variable）- 1-D LoDTensor，表示（groud-truth）是密集的。
        - **im_info** （Variable）- 是一个形为[N，3]的2-D LoDTensor。N是batch大小，第二维上的3维分别代表高度，宽度和比例(scale)
        - **rpn_batch_size_per_im** （int）- 每个图像中RPN示例总数。
        - **rpn_straddle_thresh** （float）- 通过straddle_thresh像素删除出现在图像外部的RPN anchor。
        - **rpn_fg_fraction** （float）- 为foreground（即class> 0）RoI小批量而标记的目标分数，第0类是background。
        - **rpn_positive_overlap** （float）- 对于一个正例的（anchor, gt box）对，是允许anchors和所有真实框之间最小重叠的。
        - **rpn_negative_overlap** （float）- 对于一个反例的（anchor, gt box）对，是允许anchors和所有真实框之间最大重叠的。

返回:

返回元组 (predicted_scores, predicted_location, target_label, target_bbox, bbox_inside_weight) :
   - **predicted_scores** 和 **predicted_location** 是RPN的预测结果。 **target_label** 和 **target_bbox** 分别是真实准确数据(ground-truth)。
   - **predicted_location** 是一个形为[F，4]的2D Tensor， **target_bbox** 的形与 **predicted_location** 相同，F是foreground anchors的数量。
   - **predicted_scores** 是一个shape为[F + B，1]的2D Tensor， **target_label** 的形与 **predict_scores** 的形相同，B是background anchors的数量，F和B取决于此算子的输入。
   - **Bbox_inside_weight** 标志着predicted_loction是否为fake_fg（假前景），其形为[F,4]。

返回类型：        元组(tuple)


**代码示例**

..  code-block:: python

        import paddle.fluid as fluid
        bbox_pred = fluid.layers.data(name=’bbox_pred’, shape=[100, 4],
                append_batch_size=False, dtype=’float32’)
        cls_logits = fluid.layers.data(name=’cls_logits’, shape=[100, 1],
                append_batch_size=False, dtype=’float32’)
        anchor_box = fluid.layers.data(name=’anchor_box’, shape=[20, 4],
                append_batch_size=False, dtype=’float32’)
        gt_boxes = fluid.layers.data(name=’gt_boxes’, shape=[10, 4],
                append_batch_size=False, dtype=’float32’)
        is_crowd = fluid.layers.data(name='is_crowd', shape=[1],
                    append_batch_size=False, dtype='float32')
        im_info = fluid.layers.data(name='im_infoss', shape=[1, 3],
                    append_batch_size=False, dtype='float32')
        loc_pred, score_pred, loc_target, score_target, bbox_inside_weight=
                fluid.layers.rpn_target_assign(bbox_pred, cls_logits,
                        anchor_box, anchor_var, gt_boxes, is_crowd, im_info)





