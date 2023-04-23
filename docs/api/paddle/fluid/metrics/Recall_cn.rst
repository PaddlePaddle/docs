.. _cn_api_fluid_metrics_Recall:

Recall
-------------------------------

.. py:class:: paddle.fluid.metrics.Recall(name=None)




召回率Recall（也称为敏感度）是指得到的相关实例数占相关实例总数的比例。https://en.wikipedia.org/wiki/Precision_and_recall 该类管理二分类任务的召回率。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.metrics.Recall

方法
::::::::::::
update(preds, labels)
'''''''''

使用当前mini-batch的预测结果更新召回率的计算。

**参数**

    - **preds** (numpy.array) - 当前mini-batch的预测结果，二分类sigmoid函数的输出，shape为[batch_size, 1]，数据类型为'float64'或'float32'。
    - **labels** (numpy.array) - 当前mini-batch的真实标签，输入的shape应与preds保持一致，shape为[batch_size, 1]，数据类型为'int32'或'int64'

**返回**
无



eval()
'''''''''

计算出最终的召回率。

**参数**
无

**返回**
召回率的计算结果。标量输出，float类型
**返回类型**
float















