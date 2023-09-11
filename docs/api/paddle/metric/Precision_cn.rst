.. _cn_api_paddle_metric_Precision:

Precision
-------------------------------

.. py:class:: paddle.metric.Precision()


精确率 Precision(也称为 positive predictive value，正预测值)是被预测为正样例中实际为正的比例。该类管理二分类任务的 precision 分数。

相关链接：https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

.. note::
这个 metric 只能用来评估二分类。

参数
::::::::::::

    - **name** (str，可选) – metric 实例的名字，默认是'precision'。


代码示例 1
::::::::::::

独立使用示例

COPY-FROM: paddle.metric.Precision:code-standalone-example

代码示例 2
::::::::::::

在 Model API 中的示例

COPY-FROM: paddle.metric.Precision:code-model-api-example

方法
::::::::::::
update(preds, labels, *args)
'''''''''

更新 Precision 的状态。

**参数**

    - **preds** (numpy.array | Tensor)：预测输出结果通常是 sigmoid 函数的输出，是一个数据类型为 float64 或 float32 的向量。
    - **labels** (numpy.array | Tensor)：真实标签的 shape 和：code: `preds` 相同，数据类型为 int32 或 int64。

**返回**

 无。

reset()
'''''''''

清空状态和计算结果。

**返回**

无。


accumulate()
'''''''''

累积的统计指标，计算和返回 precision 值。

**返回**

precision 值，一个标量。


name()
'''''''''

返回 Metric 实例的名字，参考上述的 name，默认是'precision'。

**返回**

评估的名字，string 类型。
