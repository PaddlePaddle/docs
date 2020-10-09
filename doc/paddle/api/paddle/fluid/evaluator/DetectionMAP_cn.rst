.. _cn_api_fluid_metrics_DetectionMAP:

DetectionMAP
-------------------------------

.. py:class:: paddle.fluid.metrics.DetectionMAP(input, gt_label, gt_box, gt_difficult=None, class_num=None, background_label=0, overlap_threshold=0.5, evaluate_difficult=True, ap_version='integral')




该OP用于计算检测网络的平均精度（mAP）。 mAP是衡量object detectors精度的指标，比如 Faster R-CNN,SSD等。它不同于召回率，它是最大精度的平均值。

通常步骤如下：

1. 根据检测器中的输入和label，计算True Positive(TP)真正例 和 False Positive(FP)假正例
2. 计算map，支持 ``11 point`` 和 ``integral`` 模式

请从以下文章中获取更多信息：
    - https://sanchom.wordpress.com/tag/average-precision/
    - https://arxiv.org/abs/1512.0232

参数：
    - **input** (Variable) – detection的输出结果，一个 shape=[M, 6] 的 LoDtensor。布局为[label, confidence, xmin, ymin, xmax, ymax],label为类别标签，confidence为置信度，xmin，ymin为检测框左上点坐标，xmax，ymax为检测框右下点坐标，数据类型为float32或float64。
    - **gt_label** (Variable) – ground truth label 的索引，它是一个形状为[N, 1]的LoDtensor，数据类型为float32或float64。
    - **gt_box** (Variable) – ground truth bounds box (bbox)，是一个具有形状的LoD张量[N, 4]。布局是[xmin, ymin, xmax, ymax]，数据类型为float32或float64。
    - **gt_difficult** (Variable|None, 可选) – 指定这个ground truth是否是一个difficult bounding bbox，它可以是一个 shape=[N, 1]的LoDTensor，也可以不被指定。默认设置为None，表示所有的ground truth标签都不是difficult bbox，数据类型为float32或float64。
    - **class_num** (int) – 检测类别的数目。
    - **background_label** (int) – 背景标签的索引，背景标签将被忽略。如果设置为-1，则所有类别将被考虑，默认为0。
    - **overlap_threshold** (float) – 判断真假阳性的阈值，默认为0.5。
    - **evaluate_difficult** (bool) – 是否考虑 difficult ground truth 进行评价，默认为 True。当 gt_difficult 为 None 时，这个参数不起作用。
    - **ap_version** (str) – 平均精度的计算方法，必须是 "integral" 或 "11point"。详情请查看 https://sanchom.wordpress.com/tag/average-precision/。 其中，11point为：11-point 插值平均精度。积分: precision-recall曲线的自然积分。

返回：变量(Variable) 计算mAP的结果，其中数据类型为float32或float64。

返回类型：变量(Variable)


**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
         
        batch_size = -1 # 可以为任意大小
        image_boxs_num = 10
        bounding_bboxes_num = 21
         
        pb = fluid.data(name='prior_box', shape=[image_boxs_num, 4],
            dtype='float32')
         
        pbv = fluid.data(name='prior_box_var', shape=[image_boxs_num, 4],
            dtype='float32')
         
        loc = fluid.data(name='target_box', shape=[batch_size, bounding_bboxes_num, 4],
            dtype='float32')
         
        scores = fluid.data(name='scores', shape=[batch_size, bounding_bboxes_num, image_boxs_num],
            dtype='float32')
         
        nmsed_outs = fluid.layers.detection_output(scores=scores,
            loc=loc, prior_box=pb, prior_box_var=pbv)
         
        gt_box = fluid.data(name="gt_box", shape=[batch_size, 4], dtype="float32")
        gt_label = fluid.data(name="gt_label", shape=[batch_size, 1], dtype="float32")
        difficult = fluid.data(name="difficult", shape=[batch_size, 1], dtype="float32")
        
        exe = fluid.Executor(fluid.CUDAPlace(0))
        map_evaluator = fluid.metrics.DetectionMAP(nmsed_outs, gt_label, gt_box, difficult, class_num = 3)
        cur_map, accum_map = map_evaluator.get_map_var()



.. py:method:: get_map_var()

返回：当前 mini-batch 的 mAP 变量和不同 mini-batch 的 mAP 累加和

.. py:method::  reset(executor, reset_program=None)

在指定的 batch 结束或者用户指定的开始时重置度量状态。

参数：
    - **executor** (Executor) – 执行reset_program的执行程序
    - **reset_program** (Program|None, 可选) – 单个program 的 reset 过程。如果设置为 None，将创建一个 program



