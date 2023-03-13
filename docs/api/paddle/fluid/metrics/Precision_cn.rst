.. _cn_api_fluid_metrics_Precision:

Precision
-------------------------------

.. py:class:: paddle.fluid.metrics.Precision(name=None)




精确率 Precision(也称为 positive predictive value，正预测值)是被预测为正样例中实际为正的比例。https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers 该类管理二分类任务的 precision 分数。


代码示例
::::::::::::


COPY-FROM: paddle.fluid.metrics.Precision

方法
::::::::::::
update(preds, labels)
'''''''''

使用当前 mini-batch 的预测结果更新精确率的计算。

**参数**

    - **preds** (numpy.array) - 当前 mini-batch 的预测结果，二分类 sigmoid 函数的输出，shape 为[batch_size, 1]，数据类型为'float64'或'float32'。
    - **labels** (numpy.array) - 当前 mini-batch 的真实标签，输入的 shape 应与 preds 保持一致，shape 为[batch_size, 1]，数据类型为'int32'或'int64'

**返回**
无



eval()
'''''''''

计算出最终的精确率。

**参数**
无

**返回**
 精确率的计算结果。标量输出，float 类型
**返回类型**
float
