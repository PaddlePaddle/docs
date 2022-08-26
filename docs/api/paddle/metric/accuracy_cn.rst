.. _cn_api_paddle_metric_accuracy:

accuracy
-------------------------------

.. py:function:: paddle.metric.accuracy(input, label, k=1, correct=None, total=None, name=None)

accuracy layer。参考 https://en.wikipedia.org/wiki/Precision_and_recall

使用输入和标签计算准确率。如果正确的标签在 topk 个预测值里，则计算结果加 1。注意：输出正确率的类型由 input 类型决定，input 和 lable 的类型可以不一样。

参数
:::::::::

    - **input** (Tensor)-数据类型为 float32,float64。输入为网络的预测值。shape 为 ``[sample_number, class_dim]`` 。
    - **label** (Tensor)-数据类型为 int64。输入为数据集的标签。shape 为 ``[sample_number, 1]`` 。
    - **k** (int，可选) - 取每个类别中 k 个预测值用于计算，默认值为 1。
    - **correct** (int，可选) - 正确预测值的个数，默认值为 None。
    - **total** (int，可选) - 总共的预测值，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

    ``Tensor``，计算出来的正确率，数据类型为 float32 的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.metric.accuracy
