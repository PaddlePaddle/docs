.. _cn_api_fluid_metrics_CompositeMetric:

CompositeMetric
-------------------------------

.. py:class:: paddle.fluid.metrics.CompositeMetric(name=None)

创建一个可以容纳若干个评价指标（如F1, accuracy, recall等）的容器，评价指标添加完成后，通过调用eval()方法可自动计算该容器内的所有评价指标。

**注意，只有输入参数列表完全相同的评价指标才可被加入到同一个CompositeMetric实例内。**

继承自：[MetricBase](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/metrics_cn.html#paddle.fluid.metrics.MetricBase)

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

向容器内添加一个新的评价指标。注意新添加的评价指标的输入参数列表必须与容器里已有的其他指标保持一致。

参数:
    - **metric** (MetricBase) – 评价指标对象，一个[MetricBase](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/metrics_cn.html#paddle.fluid.metrics.MetricBase)的实例。

返回：无


.. py:method:: update(preds, labels)

更新容器中的每个评价指标。

参数:
    - **preds**  (numpy.array) - 当前mini-batch的预测结果，输入的shape和dtype应与该容器内添加的评价指标的要求保持一致。
    - **labels**  (numpy.array) - 当前mini-batch的真实标签，输入的shape和dtype应与该容器内添加的评价指标的要求保持一致

返回：无

.. py:method:: eval()

按照添加顺序计算出各个评价指标。

参数: 无

返回: 列表存储的各个评价指标的计算结果。每个计算结果的数据类型和shape取决于被添加的评价指标的定义

返回类型: list









