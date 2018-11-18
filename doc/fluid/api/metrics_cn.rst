
.. _cn_api_fluid_metrics_CompositeMetric

CompositeMetric
>>>>>>>>>>>>

.. py:class:: class paddle.fluid.metrics.CompositeMetric(name=None)

在一个实例中组合多个指标。例如，将F1、准确率、召回率合并为一个指标。

**代码示例**

.. code-block:: python

        labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
        data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
        pred = fluid.layers.fc(input=data, size=1000, act="tanh")
        comp = fluid.metrics.CompositeMetric()
        acc = fluid.metrics.Precision()
        recall = fluid.metrics.Recall()
        comp.add_metric(acc)
        comp.add_metric(recall)
        for pass in range(PASSES):
        comp.reset()
        for data in train_reader():
            loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
        comp.update(preds=preds, labels=labels)
        numpy_acc, numpy_recall = comp.eval()

.. py::method:: add_metric(metric）

向CompositeMetric添加一个度量指标

参数:	
    - **metric** –  MetricBase的一个实例。


.. py::method:: update(preds, labels)

更新序列中的每个指标。

参数:

    - **preds** (numpy.array) -当前mini batch的预测
    - **labels** (numpy.array) -当前minibatch的label，如果标签是one-hot或soft-laebl 编码，应该自定义相应的更新规则。

.. py::method:: eval()

按顺序评估每个指标。


返回：Python中的度量值列表。
返回类型：list（float | numpy.array）


.. _cn_api_fluid_metrics_Precision

Precision
>>>>>>>>>>>>

.. py:class:: class paddle.fluid.metrics.Precision(name=None)

Precision(也称为 positive predictive value)是被预测为正样例中实际为正的比例。https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
注:二分类中，Precision与Accuracy不同。Accuracy=true positive /total instances  precision = true positive / all positive instance


**代码示例**

.. code-block:: python

    metric = fluid.metrics.Precision() 
    
    for pass in range(PASSES):
        metric.reset() 
        for data in train_reader():
        loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
         metric.update(preds=preds, labels=labels) 
        numpy_precision = metric.eval()

.. _cn_api_fluid_metrics_Recall

Recall
>>>>>>>>>>>>

.. py:class:: class paddle.fluid.metrics.Recall(name=None)

召回率（也称为敏感度）是度量有多个正例被分为正例

https://en.wikipedia.org/wiki/Precision_and_recall

**代码示例**

.. code-block:: python

        metric = fluid.metrics.Recall() 
        
        for pass in range(PASSES):
            metric.reset() 
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
                metric.update(preds=preds, labels=labels) 
                numpy_recall = metric.eval()


.. _cn_api_fluid_metrics_Accuracy

Accuracy
>>>>>>>>>>>>

class paddle.fluid.metrics.Accuracy(name=None)

累加mini-batch正确率，计算每次pass的平均准确率。https://en.wikipedia.org/wiki/Accuracy_and_precision

参数:
    - *name* ——度量标准名称

**代码示例**

.. code-block:: python

    labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
    data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
    pred = fluid.layers.fc(input=data, size=1000, act="tanh")
    minibatch_accuracy = fluid.layers.accuracy(pred, label)
    accuracy_evaluator = fluid.metrics.Accuracy()
    for pass in range(PASSES):
        accuracy_evaluator.reset()
        for data in train_reader():
            batch_size = data[0]
            loss = exe.run(fetch_list=[cost, minibatch_accuracy])
        accuracy_evaluator.update(value=minibatch_accuracy, weight=batch_size)
        numpy_acc = accuracy_evaluator.eval()


.. py:method:: update(value, weight)

更新mini batch的状态.

参数：	
    - **value** (float|numpy.array) – 每个mini batch的正确率
    - **weight** (int|float) – batch 大小

.. _cn_api_fluid_metrics_EditDistance

EditDistance
>>>>>>>>>>>>

.. py:class:: class paddle.fluid.metrics.EditDistance(name)

编辑距离是通过计算将一个字符串转换为另一个字符串所需的最小操作数来量化两个字符串(例如单词)之间的差异的一种方法。参考 https://en.wikipedia.org/wiki/Edit_distance
从mini batch中累计编辑距离和序列号，计算所有batch的平均编辑距离和实例错误。

参数:
    - **name** - 度量标准名称

**代码示例**

.. code-block:: python

    distances, seq_num = fluid.layers.edit_distance(input, label)
    distance_evaluator = fluid.metrics.EditDistance()
    for epoch in PASS_NUM:
        distance_evaluator.reset()
        for data in batches:
            loss = exe.run(fetch_list=[cost] + list(edit_distance_metrics))
        distance_evaluator.update(distances, seq_num)
        distance, instance_error = distance_evaluator.eval()

在上面的例子中：'distance'是一个pass中的编辑距离的平均值。 'instance_error'是一个pass中的实例的错误率。

.. _cn_api_fluid_metrics_DetectionMAP

DetectionMAP
>>>>>>>>>>>>

计算 detection 平均精度（mAP）。 mAP是衡量object detectors精度的指标，比如 Faster R-CNN,SSD等。它不同于召回率，它是最大精度的平均值。 请从以下文章中获取更多信息：

https://sanchom.wordpress.com/tag/average-precision/

https://arxiv.org/abs/1512.02325

通常步骤如下：

    1 根据detectors中的输入和label，计算  true positive 和 false positive
    2 计算map，支持 ‘11 point’ and ‘integral’
