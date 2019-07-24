.. _cn_api_fluid_metrics_Auc:

Auc
-------------------------------

.. py:class:: paddle.fluid.metrics.Auc(name, curve='ROC', num_thresholds=4095)

Auc度量用于二分类。参考 https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve  。请注意auc度量是用Python实现的，可能速度略慢。

auc函数创建四个局部变量true_positives, true_negatives, false_positives和false_negatives，用于计算AUC。对于离散化AUC曲线，临界值线性间隔设置以便计算召回率和准确率的值，用false positive率的召回值高度计算ROC曲线面积，用recall的准确值高度计算PR曲线面积。

参数：
    - **name** - 度量名
    - **curve** - 将要计算的曲线名的详情，曲线包括ROC（默认）或者PR（Precision-Recall-curve）。

注：目前只用Python实现ROC曲线

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    # 初始化auc度量
    auc_metric = fluid.metrics.Auc("ROC")

    # 假设batch_size为128
    batch_num = 100
    batch_size = 128

    for batch_id in range(batch_num):
        
        class0_preds = np.random.random(size = (batch_size, 1))
        class1_preds = 1 - class0_preds
         
        preds = np.concatenate((class0_preds, class1_preds), axis=1)
         
        labels = np.random.randint(2, size = (batch_size, 1))
        auc_metric.update(preds = preds, labels = labels)
        
        # 应为一个接近0.5的值，因为preds是随机指定的
        print("auc for iteration %d is %.2f" % (batch_id, auc_metric.eval()))

.. py:method:: update(preds, labels)

用给定的预测值和标签更新auc曲线。

参数： 
    - **preds** – 形状为(batch_size, 2)的numpy数组，preds[i][j]表示将实例i划分为类别j的概率。
    - **labels** – 形状为(batch_size, 1)的numpy数组，labels[i]为0或1，代表实例i的标签。


.. py:method:: eval()

返回auc曲线下的区域（一个float值）。











