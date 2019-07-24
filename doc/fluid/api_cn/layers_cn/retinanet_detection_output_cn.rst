.. _cn_api_fluid_layers_retinanet_detection_output:

retinanet_detection_output
-------------------------------

.. py:function:: paddle.fluid.layers.retinanet_detection_output(bboxes, scores, anchors, im_info, score_threshold=0.05, nms_top_k=1000, keep_top_k=100, nms_threshold=0.3, nms_eta=1.0)

**Retinanet的检测输出层**

此操作通过执行以下步骤获取检测结果：

1. 根据anchor框解码每个FPN级别的最高得分边界框预测。
2. 合并所有级别的顶级预测并对其应用多级非最大抑制（NMS）以获得最终检测。


参数：
    - **bboxes**  (List) – 来自多个FPN级别的张量列表。每个元素都是一个三维张量，形状[N，Mi，4]代表Mi边界框的预测位置。N是batch大小，Mi是第i个FPN级别的边界框数，每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。
    - **scores**  (List) – 来自多个FPN级别的张量列表。每个元素都是一个三维张量，各张量形状为[N，Mi，C]，代表预测的置信度预测。 N是batch大小，C是类编号（不包括背景），Mi是第i个FPN级别的边界框数。对于每个边界框，总共有C个评分。
    - **anchors**  (List) – 具有形状[Mi，4]的2-D Tensor表示来自所有FPN级别的Mi anchor框的位置。每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。
    - **im_info**  (Variable) – 形状为[N，3]的2-D LoDTensor表示图像信息。 N是batch大小，每个图像信息包括高度，宽度和缩放比例。
    - **score_threshold**  (float) – 用置信度分数剔除边界框的过滤阈值。
    - **nms_top_k**  (int) – 根据NMS之前的置信度保留每个FPN层的最大检测数。
    - **keep_top_k**  (int) – NMS步骤后每个图像要保留的总边界框数。 -1表示在NMS步骤之后保留所有边界框。
    - **nms_threshold**  (float) – NMS中使用的阈值.
    - **nms_eta**  (float) – adaptive NMS的参数.



返回：
检测输出是具有形状[No，6]的LoDTensor。 每行有六个值：[标签，置信度，xmin，ymin，xmax，ymax]。 No是此mini batch中的检测总数。 对于每个实例，第一维中的偏移称为LoD，偏移值为N + 1，N是batch大小。 第i个图像具有LoD [i + 1]  -  LoD [i]检测结果，如果为0，则第i个图像没有检测到结果。 如果所有图像都没有检测到结果，则LoD将设置为0，输出张量为空（None）。


返回类型：变量（Variable）

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid

  bboxes = layers.data(name='bboxes', shape=[1, 21, 4],
      append_batch_size=False, dtype='float32')
  scores = layers.data(name='scores', shape=[1, 21, 10],
      append_batch_size=False, dtype='float32')
  anchors = layers.data(name='anchors', shape=[21, 4],
      append_batch_size=False, dtype='float32')
  im_info = layers.data(name="im_info", shape=[1, 3],
      append_batch_size=False, dtype='float32')
  nmsed_outs = fluid.layers.retinanet_detection_output(
                                          bboxes=[bboxes, bboxes],
                                          scores=[scores, scores],
                                          anchors=[anchors, anchors],
                                          im_info=im_info,
                                          score_threshold=0.05,
                                          nms_top_k=1000,
                                          keep_top_k=100,
                                          nms_threshold=0.3,
                                          nms_eta=1.)



