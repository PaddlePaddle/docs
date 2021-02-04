.. _cn_api_paddle_metric_accuracy:

accuracy
-------------------------------

.. py:function:: paddle.metric.accuracy(input, label, k=1, correct=None, total=None, name=None)

accuracy layer。 参考 https://en.wikipedia.org/wiki/Precision_and_recall

使用输入和标签计算准确率。 如果正确的标签在topk个预测值里，则计算结果加1。注意：输出正确率的类型由input类型决定，input和lable的类型可以不一样。

参数
:::::::::

    - **input** (Tensor)-数据类型为float32,float64。输入为网络的预测值。shape为 ``[sample_number, class_dim]`` 。
    - **label** (Tensor)-数据类型为int64，int32。输入为数据集的标签。shape为 ``[sample_number, 1]`` 。
    - **k** (int64|int32，可选) - 取每个类别中k个预测值用于计算，默认值为1。
    - **correct** (int64|int32, 可选)-正确预测值的个数，默认值为None。
    - **total** (int64|int32，可选)-总共的预测值，默认值为None。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::

    ``Tensor``，计算出来的正确率，数据类型为float32的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle

    predictions = paddle.to_tensor([[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]], dtype='float32')
    label = paddle.to_tensor([[2], [0]], dtype="int64")
    result = paddle.metric.accuracy(input=predictions, label=label, k=1)
    # [0.5]
