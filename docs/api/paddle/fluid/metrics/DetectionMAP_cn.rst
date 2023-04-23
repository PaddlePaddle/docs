.. _cn_api_fluid_metrics_DetectionMAP:

DetectionMAP
-------------------------------

.. py:class:: paddle.fluid.metrics.DetectionMAP(input, gt_label, gt_box, gt_difficult=None, class_num=None, background_label=0, overlap_threshold=0.5, evaluate_difficult=True, ap_version='integral')




该 OP 用于计算检测网络的平均精度（mAP）。 mAP 是衡量 object detectors 精度的指标，比如 Faster R-CNN,SSD 等。它不同于召回率，它是最大精度的平均值。

通常步骤如下：

1. 根据检测器中的输入和 label，计算 True Positive(TP)真正例 和 False Positive(FP)假正例
2. 计算 map，支持 ``11 point`` 和 ``integral`` 模式

请从以下文章中获取更多信息：
    - https://sanchom.wordpress.com/tag/average-precision/
    - https://arxiv.org/abs/1512.0232

参数
::::::::::::

    - **input** (Variable) – detection 的输出结果，一个 shape=[M, 6] 的 LoDtensor。布局为[label, confidence, xmin, ymin, xmax, ymax],label 为类别标签，confidence 为置信度，xmin，ymin 为检测框左上点坐标，xmax，ymax 为检测框右下点坐标，数据类型为 float32 或 float64。
    - **gt_label** (Variable) – ground truth label 的索引，它是一个形状为[N, 1]的 LoDtensor，数据类型为 float32 或 float64。
    - **gt_box** (Variable) – ground truth bounds box (bbox)，是一个具有形状的 LoDTensor[N, 4]。布局是[xmin, ymin, xmax, ymax]，数据类型为 float32 或 float64。
    - **gt_difficult** (Variable|None，可选) – 指定这个 ground truth 是否是一个 difficult bounding bbox，它可以是一个 shape=[N, 1]的 LoDTensor，也可以不被指定。默认设置为 None，表示所有的 ground truth 标签都不是 difficult bbox，数据类型为 float32 或 float64。
    - **class_num** (int) – 检测类别的数目。
    - **background_label** (int) – 背景标签的索引，背景标签将被忽略。如果设置为-1，则所有类别将被考虑，默认为 0。
    - **overlap_threshold** (float) – 判断真假阳性的阈值，默认为 0.5。
    - **evaluate_difficult** (bool) – 是否考虑 difficult ground truth 进行评价，默认为 True。当 gt_difficult 为 None 时，这个参数不起作用。
    - **ap_version** (str) – 平均精度的计算方法，必须是 "integral" 或 "11point"。详情请查看 https://sanchom.wordpress.com/tag/average-precision/。其中，11point 为：11-point 插值平均精度。积分：precision-recall 曲线的自然积分。

返回
::::::::::::
变量(Variable) 计算 mAP 的结果，其中数据类型为 float32 或 float64。

返回类型
::::::::::::
变量(Variable)


代码示例
::::::::::::


COPY-FROM: paddle.fluid.metrics.DetectionMAP

方法
::::::::::::
get_map_var()
'''''''''

**返回**
当前 mini-batch 的 mAP 变量和不同 mini-batch 的 mAP 累加和

reset(executor, reset_program=None)
'''''''''

在指定的 batch 结束或者用户指定的开始时重置度量状态。

**参数**

    - **executor** (Executor) – 执行 reset_program 的执行程序
    - **reset_program** (Program|None，可选) – 单个 program 的 reset 过程。如果设置为 None，将创建一个 program
