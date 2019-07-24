.. _cn_api_fluid_metrics_CompositeMetric:

CompositeMetric
-------------------------------

.. py:class:: paddle.fluid.metrics.CompositeMetric(name=None)

在一个实例中组合多个指标。例如，将F1、准确率、召回率合并为一个指标。

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
                 [0.2], [0.3], [0.5], [0.8], [0.6]]
        labels = [[0], [1], [1], [1], [1],
                  [0], [0], [0], [0], [0]]
        preds = np.array(preds)
        labels = np.array(labels)

        comp = fluid.metrics.CompositeMetric()
        precision = fluid.metrics.Precision()
        recall = fluid.metrics.Recall()
        comp.add_metric(precision)
        comp.add_metric(recall)
        
        comp.update(preds=preds, labels=labels)
        numpy_precision, numpy_recall = comp.eval()
        print("expect precision: %.2f, got %.2f" % ( 3. / 5, numpy_precision ) )
        print("expect recall: %.2f, got %.2f" % (3. / 4, numpy_recall ) )


.. py:method:: add_metric(metric)

向CompositeMetric添加一个度量指标

参数:
    - **metric** –  MetricBase的一个实例。



.. py:method:: update(preds, labels)

更新序列中的每个指标。

参数:
    - **preds**  (numpy.array) - 当前mini batch的预测
    - **labels**  (numpy.array) - 当前minibatch的label，如果标签是one-hot或soft-laebl 编码，应该自定义相应的更新规则。

.. py:method:: eval()

按顺序评估每个指标。


返回：Python中的度量值列表。

返回类型：list（float | numpy.array）








