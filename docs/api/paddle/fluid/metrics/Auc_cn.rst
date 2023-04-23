.. _cn_api_fluid_metrics_Auc:

Auc
-------------------------------
.. py:class:: paddle.fluid.metrics.Auc(name, curve='ROC', num_thresholds=4095)




**注意**：目前只用Python实现Auc，可能速度略慢

该接口计算Auc，在二分类(binary classification)中广泛使用。相关定义参考 https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 。

该接口创建四个局部变量true_positives, true_negatives, false_positives和false_negatives，用于计算Auc。为了离散化AUC曲线，使用临界值的线性间隔来计算召回率和准确率的值。用false positive的召回值高度计算ROC曲线面积，用recall的准确值高度计算PR曲线面积。

参数
::::::::::::

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **curve** (str) - 将要计算的曲线名的详情，曲线包括ROC（默认）或者PR（Precision-Recall-curve）。

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

用给定的预测值和标签更新Auc曲线。

**参数**
 
    - **preds** (numpy.array) - 维度为[batch_size, 2]，preds[i][j]表示将实例i划分为类别j的概率。
    - **labels** (numpy.array) - 维度为[batch_size, 1]，labels[i]为0或1，代表实例i的标签。

**返回**
无

eval()
'''''''''

该函数计算并返回Auc值。

**返回**
Auc值

**返回类型**
float

