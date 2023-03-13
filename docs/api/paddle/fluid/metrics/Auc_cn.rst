.. _cn_api_fluid_metrics_Auc:

Auc
-------------------------------
.. py:class:: paddle.fluid.metrics.Auc(name, curve='ROC', num_thresholds=4095)




**注意**：目前只用 Python 实现 Auc，可能速度略慢

该接口计算 Auc，在二分类(binary classification)中广泛使用。相关定义参考 https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 。

该接口创建四个局部变量 true_positives, true_negatives, false_positives 和 false_negatives，用于计算 Auc。为了离散化 AUC 曲线，使用临界值的线性间隔来计算召回率和准确率的值。用 false positive 的召回值高度计算 ROC 曲线面积，用 recall 的准确值高度计算 PR 曲线面积。

参数
::::::::::::

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **curve** (str) - 将要计算的曲线名的详情，曲线包括 ROC（默认）或者 PR（Precision-Recall-curve）。

返回
::::::::::::
初始化后的 ``Auc`` 对象

返回类型
::::::::::::
Auc

代码示例
::::::::::::


COPY-FROM: paddle.fluid.metrics.Auc

方法
::::::::::::
update(preds, labels)
'''''''''

用给定的预测值和标签更新 Auc 曲线。

**参数**

    - **preds** (numpy.array) - 维度为[batch_size, 2]，preds[i][j]表示将实例 i 划分为类别 j 的概率。
    - **labels** (numpy.array) - 维度为[batch_size, 1]，labels[i]为 0 或 1，代表实例 i 的标签。

**返回**
无

eval()
'''''''''

该函数计算并返回 Auc 值。

**返回**
Auc 值

**返回类型**
float
