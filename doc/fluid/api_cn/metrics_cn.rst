#################
 fluid.metrics
#################



.. _cn_api_fluid_metrics_Accuracy:

Accuracy
-------------------------------

.. py:class:: paddle.fluid.metrics.Accuracy(name=None)

累加mini-batch正确率，计算每次pass的平均准确率。https://en.wikipedia.org/wiki/Accuracy_and_precision

参数:
    - **name** — 度量标准的名称

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







.. _cn_api_fluid_metrics_Auc:

Auc
-------------------------------

.. py:class:: paddle.fluid.metrics.Auc(name, curve='ROC', num_thresholds=4095)

Auc度量适用于二分类。参考 https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve  。需要注意auc度量本身是用Python计算值。如果关心速度，请用fluid.layers.auc。

auc函数创建四个局部变量true_positives, true_negatives, false_positives和false_negatives，用于计算AUC。对于离散化AUC曲线，临界值线性间隔设置以便计算召回率和准确率的值，用false positive率的召回值高度计算ROC曲线面积，用recall的准确值高度计算PR曲线面积。

参数：
    - **name** - 度量名
    - **curve** - 将要计算的曲线名的详情，曲线包括ROC（默认）或者PR（Precision-Recall-curve）。

注：目前只用Python实现ROC曲线

**代码示例**：

.. code-block:: python

    pred = fluid.layers.fc(input=data, size=1000, act="tanh")
    metric = fluid.metrics.Auc()
    for data in train_reader():
        loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
        metric.update(preds, labels)
        numpy_auc = metric.eval()














.. _cn_api_fluid_metrics_ChunkEvaluator:

ChunkEvaluator
-------------------------------

.. py:class:: paddle.fluid.metrics.ChunkEvaluator(name=None)

用mini-batch的chunk_eval累计counter numbers，用累积的counter numbers计算准确率、召回率和F1值。对于chunking的基础知识，请参考 .. _Chunking with Support Vector Machines: https://aclanthology.info/pdf/N/N01/N01-1025.pdf 。ChunkEvalEvaluator计算块检测（chunk detection）的准确率，召回率和F1值，支持IOB, IOE, IOBES和IO标注方案。

**代码示例**：

.. code-block:: python

        labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
        data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
        pred = fluid.layers.fc(input=data, size=1000, act="tanh")
        precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks = layers.chunk_eval(
        input=pred,
        label=label)
        metric = fluid.metrics.ChunkEvaluator()
        for data in train_reader():
            loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
            metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
            numpy_precision, numpy_recall, numpy_f1 = metric.eval()
    
.. py:method:: update(num_infer_chunks, num_label_chunks, num_correct_chunks)

基于layers.chunk_eval()输出更新状态（state)输出

参数:
    - **num_infer_chunks** (int|numpy.array): 给定minibatch的Interface块数。
    - **num_label_chunks** (int|numpy.array): 给定minibatch的Label块数。
    - **num_correct_chunks** （int|numpy.array）: 给定minibatch的Interface和Label的块数







.. _cn_api_fluid_metrics_CompositeMetric:

CompositeMetric
-------------------------------

.. py:class:: paddle.fluid.metrics.CompositeMetric(name=None)

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








.. _cn_api_fluid_metrics_DetectionMAP:

DetectionMAP
-------------------------------

.. py:class:: paddle.fluid.metrics.DetectionMAP(name=None)

计算 detection 平均精度（mAP）。 mAP是衡量object detectors精度的指标，比如 Faster R-CNN,SSD等。它不同于召回率，它是最大精度的平均值。 请从以下文章中获取更多信息：

https://sanchom.wordpress.com/tag/average-precision/

https://arxiv.org/abs/1512.02325

通常步骤如下：

1. 根据detectors中的输入和label，计算  true positive 和 false positive
2. 计算map，支持 ‘11 point’ and ‘integral’

参数：
	- **input** (Variable) – detection的结果，一个 shape=[M, 6] 的 lodtensor。布局为[label, confidence, xmin, ymin, xmax, ymax]
	- **gt_label** (Variable) – ground truth label 的索引，它是一个形状为[N, 1]的lodtensor
	- **gt_box** (Variable) – ground truth bounds box (bbox)，是一个具有形状的lod张量[N, 4]。布局是[xmin, ymin, xmax, ymax]
	- **gt_difficult** (Variable|None) – 指定这个ground truth是否是一个difficult bounding bbox，它可以是一个 shape=[N, 1]的LoDTensor，也可以不被指定。如果设置为None，则表示所有的ground truth标签都不是difficult bbox。
	- **class_num** (int) – 检测类别的数目
	- **background_label** (int) – 背景标签的索引，背景标签将被忽略。如果设置为-1，则所有类别将被考虑，默认为0。
	- **overlap_threshold** (float) – 判断真假阳性的阈值，默认为0.5
	- **evaluate_difficult** (bool) – 是否考虑 difficult ground truth 进行评价，默认为 True。当 gt_difficult 为 None 时，这个参数不起作用。
	- **ap_version** (string) – 平均精度的计算方法，必须是 "integral" 或 "11point"。详情请查看 https://sanchom.wordpress.com/tag/averageprecision/。 其中，11point为：11-point 插值平均精度。积分: precision-recall曲线的自然积分。

**代码示例**

.. code-block:: python

	exe = fluid.Executor(place)
	map_evaluator = fluid.Evaluator.DetectionMAP(input,
	    gt_label, gt_box, gt_difficult)
	cur_map, accum_map = map_evaluator.get_map_var()
	fetch = [cost, cur_map, accum_map]
	for epoch in PASS_NUM:
	    map_evaluator.reset(exe)
	    for data in batches:
	        loss, cur_map_v, accum_map_v = exe.run(fetch_list=fetch)



在上述例子中：
	
	"cur_map_v" 是当前 mini-batch 的 mAP
	
	"accum_map_v" 是一个 pass 的 mAP累加和

.. py:method:: get_map_var()

返回：当前 mini-batch 的 mAP 变量，和跨 mini-batch 的 mAP 累加和

.. py:methord::  reset(executor, reset_program=None)

在指定 batch 的每一 pass/user  开始时重置度量状态。

参数：
	- **executor** (Executor) – 执行reset_program的执行程序
	- **reset_program** (Program|None) –  单一 program 的 reset 过程。如果设置为 None，将创建一个 program



.. _cn_api_fluid_metrics_EditDistance:

EditDistance
-------------------------------

.. py:class:: paddle.fluid.metrics.EditDistance(name)

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







.. _cn_api_fluid_metrics_MetricBase:

MetricBase
-------------------------------

.. py:class:: paddle.fluid.metrics.MetricBase(name)

所有Metrics的基类。MetricBase为模型估计方法定义一组接口。Metrics累积连续的两个minibatch之间的度量状态，对每个minibatch用最新接口将当前minibatch值添加到全局状态。用eval函数来计算last reset()或者scratch on()中累积的度量值。如果需要定制一个新的metric，请继承自MetricBase和自定义实现类。

参数：
    - **name** (str) - metric实例名。例如准确率（accuracy）。如果想区分一个模型里不同的metrics，则需要实例名。

.. py:method:: reset()

        reset()清除度量（metric）的状态（state）。默认情况下，状态（state）包含没有 ``_`` 前缀的metric。reset将这些状态设置为初始状态。如果不想使用隐式命名规则，请自定义reset接口。

.. py:method:: get_config()

获取度量（metric)状态和当前状态。状态（state）包含没有 ``_`` 前缀的成员。
        
参数：**None**

返回：metric对应到state的字典

返回类型：字典（dict）


.. py:method:: update(preds,labels)

更新每个minibatch的度量状态（metric states），用户可通过Python或者C++操作符计算minibatch度量值（metric）。

参数：
     - **preds** (numpy.array) - 当前minibatch的预测
     - **labels** (numpy.array) - 当前minibatch的标签，如果标签为one-hot或者soft-label，应该自定义相应的更新规则。

.. py:method:: eval()

基于累积状态（accumulated states）评估当前度量（current metric）。

返回：metrics（Python中）

返回类型：float|list(float)|numpy.array







.. _cn_api_fluid_metrics_Precision:

Precision
-------------------------------

.. py:class:: paddle.fluid.metrics.Precision(name=None)

Precision(也称为 positive predictive value,正预测值)是被预测为正样例中实际为正的比例。https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
注:二分类中，Precision与Accuracy不同,

.. math::
    Accuracy  & = \frac{true \quad positive}{total \quad instances(所有样例)}  \\\\
    Precision & = \frac{true \quad positive}{all \quad positive \quad instances(所有正样例)}


**代码示例**

.. code-block:: python

    metric = fluid.metrics.Precision() 
    
    for pass in range(PASSES):
        metric.reset() 
        for data in train_reader():
        loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
         metric.update(preds=preds, labels=labels) 
        numpy_precision = metric.eval()







.. _cn_api_fluid_metrics_Recall:

Recall
-------------------------------

.. py:class:: paddle.fluid.metrics.Recall(name=None)

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








