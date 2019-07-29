.. _cn_api_fluid_metrics_Precision:

Precision
-------------------------------

.. py:class:: paddle.fluid.metrics.Precision(name=None)

Precision(也称为 positive predictive value,正预测值)是被预测为正样例中实际为正的比例。
https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
该类管理二分类任务的precision分数。



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
    numpy_precision = metric.eval()
    
    print("expct precision: %.2f and got %.2f" % ( 3.0 / 5.0, numpy_precision))







