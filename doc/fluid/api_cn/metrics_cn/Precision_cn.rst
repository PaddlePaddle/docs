.. _cn_api_fluid_metrics_Precision:

Precision
-------------------------------

.. py:class:: paddle.fluid.metrics.Precision(name=None)




精确率Precision(也称为 positive predictive value,正预测值)是被预测为正样例中实际为正的比例。 https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers 该类管理二分类任务的precision分数。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    metric = fluid.metrics.Precision() 

    # 生成预测值和标签

    preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
             [0.2], [0.3], [0.5], [0.8], [0.6]]
             
    labels = [[0], [1], [1], [1], [1],
              [0], [0], [0], [0], [0]]
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    metric.update(preds=preds, labels=labels) 
    precision = metric.eval()
    
    print("expected precision: %.2f and got %.2f" % ( 3.0 / 5.0, precision))



.. py:method:: update(preds, labels)

使用当前mini-batch的预测结果更新精确率的计算。

参数: 
    - **preds** (numpy.array) - 当前mini-batch的预测结果，二分类sigmoid函数的输出，shape为[batch_size, 1]，数据类型为'float64'或'float32'。
    - **labels** (numpy.array) - 当前mini-batch的真实标签，输入的shape应与preds保持一致，shape为[batch_size, 1]，数据类型为'int32'或'int64'

返回：无



.. py:method:: eval()

计算出最终的精确率。

参数：无

返回: 精确率的计算结果。标量输出，float类型
返回类型：float


