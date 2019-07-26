.. _cn_api_fluid_layers_detection_output:

detection_output
-------------------------------

.. py:function:: paddle.fluid.layers.detection_output(loc, scores, prior_box, prior_box_var, background_label=0, nms_threshold=0.3, nms_top_k=400, keep_top_k=200, score_threshold=0.01, nms_eta=1.0)

Detection Output Layer for Single Shot Multibox Detector(SSD)

该操作符用于获得检测结果，执行步骤如下：

    1.根据prior box框解码输入边界框（bounding box）预测

    2.通过运用多类非极大值抑制(NMS)获得最终检测结果

请注意，该操作符不将最终输出边界框剪切至图像窗口。

参数：
    - **loc** (Variable) - 一个三维张量（Tensor），维度为[N,M,4]，代表M个bounding bboxes的预测位置。N是批尺寸，每个边界框（boungding box）有四个坐标值，布局为[xmin,ymin,xmax,ymax]
    - **scores** (Variable) - 一个三维张量（Tensor），维度为[N,M,C]，代表预测置信预测。N是批尺寸，C是类别数，M是边界框数。对每类一共M个分数，对应M个边界框
    - **prior_box** (Variable) - 一个二维张量（Tensor),维度为[M,4]，存储M个框，每个框代表[xmin,ymin,xmax,ymax]，[xmin,ymin]是anchor box的左上坐标，如果输入是图像特征图，靠近坐标系统的原点。[xmax,ymax]是anchor box的右下坐标
    - **prior_box_var** (Variable) - 一个二维张量（Tensor），维度为[M,4]，存有M变量群
    - **background_label** (float) - 背景标签索引，背景标签将会忽略。若设为-1，将考虑所有类别
    - **nms_threshold** (int) - 用于NMS的临界值（threshold）
    - **nms_top_k** (int) - 基于score_threshold过滤检测后，根据置信数维持的最大检测数
    - **keep_top_k** (int) - NMS步后，每一图像要维持的总bbox数
    - **score_threshold** (float) - 临界函数（Threshold），用来过滤带有低置信数的边界框（bounding box）。若未提供，则考虑所有框
    - **nms_eta** (float) - 适应NMS的参数

返回：
  输出一个LoDTensor，形为[No,6]。每行有6个值：[label,confidence,xmin,ymin,xmax,ymax]。No是该mini-batch的总检测数。对每个实例，第一维偏移称为LoD，偏移数为N+1，N是batch size。第i个图像有LoD[i+1]-LoD[i]检测结果。如果为0，第i个图像无检测结果。如果所有图像都没有检测结果，LoD会被设置为{1}，并且输出张量只包含一个值-1。（1.3版本后对于没有检测结果的boxes, LoD的值由之前的{0}调整为{1}）

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    
    import paddle.fluid as fluid
    pb = fluid.layers.data(name='prior_box', shape=[10, 4],
             append_batch_size=False, dtype='float32')
    pbv = fluid.layers.data(name='prior_box_var', shape=[10, 4],
              append_batch_size=False, dtype='float32')
    loc = fluid.layers.data(name='target_box', shape=[2, 21, 4],
              append_batch_size=False, dtype='float32')
    scores = fluid.layers.data(name='scores', shape=[2, 21, 10],
              append_batch_size=False, dtype='float32')
    nmsed_outs = fluid.layers.detection_output(scores=scores,
                           loc=loc,
                           prior_box=pb,
                           prior_box_var=pbv)






