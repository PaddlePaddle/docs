.. _cn_api_paddle_metric_accuracy:

accuracy
-------------------------------

.. py:function:: paddle.metric.accuracy(input, label, k=1, correct=None, total=None, name=None)

accuracy layer。参考 https://en.wikipedia.org/wiki/Precision_and_recall

使用 input 和 label 计算准确率。如果正确的 label 在 top k 个预测值里，则计算结果加 1。

.. note::
    输出正确率的类型由 input 的类型决定，input 和 label 的类型可以不一样。

参数
:::::::::

    - **input** (Tensor)-数据类型为 float32,float64 的 Tensor。accuracy layer 的输入，即网络的预测值。shape 为 ``[sample_number, class_dim]`` 。
    - **label** (Tensor)-数据类型为 int64,int32 的 Tensor。数据集的标签。shape 为 ``[sample_number, 1]`` 。
    - **k** (int，可选) - 数据类型为 int64,int32。取每个类别中 top k 个预测值用于计算，默认值为 1。
    - **correct** (Tensor，可选) - 数据类型为 int64,int32 的 Tensor。正确预测值的个数，默认值为 None。
    - **total** (Tensor，可选) - 数据类型为 int64,int32 的 Tensor。总共的预测值，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

    ``Tensor``，计算出来的正确率，数据类型为 float32 的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.metric.accuracy
