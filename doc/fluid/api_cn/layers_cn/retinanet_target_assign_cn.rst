.. _cn_api_fluid_layers_retinanet_target_assign:

retinanet_target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.retinanet_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, gt_labels, is_crowd, im_info, num_classes=1, positive_overlap=0.5, negative_overlap=0.4)

**Retinanet的目标分配层**

对于给定anchors和真实(ground-truth)框之间的Intersection-over-Union（IoU）重叠，该层可以为每个anchor分配分类和回归目标，同时这些目标标签用于训练Retinanet。每个anchor都分配有长度为num_classes的一个one-hot分类目标向量，以及一个4向量的框回归目标。分配规则如下：

1.在以下情况下，anchor被分配到真实框：
（i）它与真实框具有最高的IoU重叠，或者（ii）与任何真实框具有高于positive_overlap（0.5）的IoU重叠。

2.对于所有真实框，当其IoU比率低于negative_overlap（0.4）时，将anchor点分配给背景。

当为锚点分配了第i个类别的真实框时，其C向量目标中的第i项设置为1，所有其他条目设置为0.当anchor被分配支背景时，所有项都设置为0。未被分配的锚点不会影响训练目标。回归目标是与指定anchor相关联的已编码真实框。



参数：
    - **bbox_pred**  (Variable) – 具有形状[N，M，4]的3-D张量表示M个边界框(bounding box)的预测位置。 N是batch大小，每个边界框有四个坐标值，为[xmin，ymin，xmax，ymax]。
    - **cls_logits**  (Variable) – 具有形状[N，M，C]的3-D张量，表示预测的置信度。 N是batch大小，C是类别的数量（不包括背景），M是边界框的数量。
    - **anchor_box**  (Variable) – 具有形状[M，4]的2-D张量，存有M个框，每个框表示为[xmin，ymin，xmax，ymax]，[xmin，ymin]是anchor的左上顶部坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是anchor的右下坐标。
    - **anchor_var**  (Variable) – 具有形状[M，4]的2-D张量，存有anchor的扩展方差。
    - **gt_boxes**  (Variable) – 真实框是具有形状[Ng，4]的2D LoDTensor，Ng是mini batch中真实框的总数。
    - **gt_labels**  (variable) – 真实值标签是具有形状[Ng，1]的2D LoDTensor，Ng是mini batch输入真实值标签的总数。
    - **is_crowd**  (Variable) – 1-D LoDTensor，标志真实值是聚群。
    - **im_info**  (Variable) – 具有形状[N，3]的2-D LoDTensor。 N是batch大小，3分别为高度，宽度和比例。
    - **num_classes**  (int32) – 种类数量。
    - **positive_overlap**  (float) – 判定（anchor，gt框）对是一个正例的anchor和真实框之间最小重叠阀值。
    - **negative_overlap**  (float) – （锚点，gt框）对是负例时anchor和真实框之间允许的最大重叠阈值。


返回：
返回元组（predict_scores，predict_location，target_label，target_bbox，bbox_inside_weight，fg_num）。 predict_scores和predict_location是Retinanet的预测结果。target_label和target_bbox为真实值。 predict_location是形为[F，4]的2D张量，target_bbox的形状与predict_location的形状相同，F是前景anchor的数量。 predict_scores是具有形状[F + B，C]的2D张量，target_label的形状是[F + B，1]，B是背景anchor的数量，F和B取决于此算子的输入。 Bbox_inside_weight标志预测位置是否为假前景，形状为[F，4]。 Fg_num是focal loss所需的前景数（包括假前景）。


返回类型：tuple

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    bbox_pred = layers.data(name='bbox_pred', shape=[1, 100, 4],
                      append_batch_size=False, dtype='float32')
    cls_logits = layers.data(name='cls_logits', shape=[1, 100, 10],
                      append_batch_size=False, dtype='float32')
    anchor_box = layers.data(name='anchor_box', shape=[100, 4],
                      append_batch_size=False, dtype='float32')
    anchor_var = layers.data(name='anchor_var', shape=[100, 4],
                      append_batch_size=False, dtype='float32')
    gt_boxes = layers.data(name='gt_boxes', shape=[10, 4],
                      append_batch_size=False, dtype='float32')
    gt_labels = layers.data(name='gt_labels', shape=[10, 1],
                      append_batch_size=False, dtype='float32')
    is_crowd = fluid.layers.data(name='is_crowd', shape=[1],
                      append_batch_size=False, dtype='float32')
    im_info = fluid.layers.data(name='im_infoss', shape=[1, 3],
                      append_batch_size=False, dtype='float32')
    loc_pred, score_pred, loc_target, score_target, bbox_inside_weight, fg_num =
          fluid.layers.retinanet_target_assign(bbox_pred, cls_logits, anchor_box,
          anchor_var, gt_boxes, gt_labels, is_crowd, im_info, 10)










