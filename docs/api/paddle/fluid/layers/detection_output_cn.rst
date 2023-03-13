.. _cn_api_fluid_layers_detection_output:

detection_output
-------------------------------

.. py:function:: paddle.fluid.layers.detection_output(loc, scores, prior_box, prior_box_var, background_label=0, nms_threshold=0.3, nms_top_k=400, keep_top_k=200, score_threshold=0.01, nms_eta=1.0)




给定回归位置偏移、置信度以及先验框信息计算检测的输出，执行步骤如下：

    1. 根据先验框(``prior_box``)信息和回归位置偏移解码出预测框坐标。

    2. 通过多类非极大值抑制(NMS)获得最终检测输出。

请注意，该操作符没有将最终输出边界框 clip 至图像大小。

参数
::::::::::::

    - **loc** (Variable) - 3-D Tensor，数据类型为 float32 或 float64，表示回归位置偏移。维度为[N,M,4]，M 是输入的预测 bounding box 的个数，N 是 batch size，每个 bounding box 有四个坐标值，格式为[xmin,ymin,xmax,ymax]，[xmin,ymin]是左上角坐标，[xmax,ymax]是右下角坐标。
    - **scores** (Variable) - 3-D Tensor，数据类型为 float32 或 float64，表示未归一化的置信度。维度为[N,M,C]，N 和 M 的含义同上，C 是类别数。
    - **prior_box** (Variable) - 2-D Tensor，表示先验框。维度为[M,4]，M 是提取的先验框个数，格式为[xmin,ymin,xmax,ymax]。
    - **prior_box_var** (Variable) - 2-D Tensor，表示先验框的方差，和 ``prior_box`` 维度相同。
    - **background_label** (int) - 背景标签类别值，背景标签类别上不做 NMS。若设为-1，将考虑所有类别。默认值是 0。
    - **nms_threshold** (float) - 用于 NMS 的阈值（threshold），默认值是 0.3。
    - **nms_top_k** (int) - 基于 score_threshold 过滤预测框后，NMS 操作前，要挑选出的置信度高的预测框的个数。默认值是 400。
    - **keep_top_k** (int) - NMS 操作后，要挑选的 bounding box 总数。默认值是 200。
    - **score_threshold** (float) - 置信度得分阈值（Threshold），在 NMS 之前用来过滤低置信数的边界框（bounding box）。若未提供，则考虑所有框。默认值是 0.001。
    - **nms_eta** (float) - 一种 adaptive NMS 的参数，仅当该值小于 1.0 时才起作用。默认值是 1.0。

返回
::::::::::::

  输出是 2-D LoDTensor，形状为[No,6]。每行有 6 个值：[label,confidence,xmin,ymin,xmax,ymax]。No 是该 mini-batch 总的检测框数。LoD 的层级数为 1，如果采用偏移的 LoD 表示，则第 i 个图像有 ``LoD[i+1] - LoD[i]`` 个检测结果，如果等于 0，则表示无检测结果。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.detection_output
