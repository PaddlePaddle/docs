.. _cn_api_fluid_metrics_Recall:

Recall
-------------------------------

.. py:class:: paddle.fluid.metrics.Recall(name=None)

召回率（也称为敏感度）是指得到的相关实例数占相关实例总数的比重

https://en.wikipedia.org/wiki/Precision_and_recall

该类管理二分类任务的召回率。

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        metric = fluid.metrics.Recall()
        # 生成预测值和标签
        preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
                 [0.2], [0.3], [0.5], [0.8], [0.6]]
        labels = [[0], [1], [1], [1], [1],
                  [0], [0], [0], [0], [0]]

        preds = np.array(preds)
        labels = np.array(labels)

        metric.update(preds=preds, labels=labels) 
        numpy_precision = metric.eval()

        print("expct precision: %.2f and got %.2f" % ( 3.0 / 4.0, numpy_precision))









