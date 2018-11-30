.. _cn_api_fluid_merics_MetricBase:

MetricBase
>>>>>>>>>>>>

.. py:class:: class paddle.fluid.metrics.MetricBase(name)

所有Metrics的基类。MetricBase为模型估计方法定义一组接口。Metrics累积连续的两个minibatch之间的度量状态，对每个minibatch用最新接口将当前minibatch值添加到全局状态。用eval函数来计算last reset()或者scratch on()中累积的度量值。如果需要定制一个新的metric，请继承自MetricBase和自定义实现类。

参数：
    - **name** (str) - metric实例名。例如准确率（accuracy）。如果想区分一个模型里不同的metrics，则需要实例名。

::


    ``reset()``

        reset()清除度量（metric）的状态（state）。默认情况下，状态（state）包含没有“_”前缀的metric。reset将这些状态设置为初始状态。如果不想使用隐式命名规则，请自定义reset接口。
    
    ``get_config()``

        获取度量（metric)状态和当前状态。状态（state）包含没有“_”前缀的成员。
        
        参数：**None**

        返回：metric对应到state的字典

        返回类型：字典（dict）

    ``update(preds,labels)``

        更新每个minibatch的度量状态（metric states），用户可通过Python或者C++操作符计算minibatch度量值（metric）。

        参数：
            - **preds** (numpy.array) - 当前minibatch的预测
            - **labels** (numpy.array) - 当前minibatch的标签，如果标签为one-hot或者soft-label，应该自定义相应的更新规则。

    ``eval()``

        基于累积状态（accumulated states）评估当前度量（current metric）。

        返回：metrics（Python中）

        返回类型：float|list(float)|numpy.array

.. _cn_api_fluid_metrics_ChunkEvaluator:

ChunkEvaluator
>>>>>>>>>>>>>>>>

.. py:class:: class paddle.fluid.metrics.ChunkEvaluator(name=None)

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
    
``update(num_infer_chunks, num_label_chunks, num_correct_chunks)``

    基于layers.chunk_eval()输出更新状态（state)输出

    参数:
        - **num_infer_chunks** (int|numpy.array): 给定minibatch的Interface块数。
        - **num_label_chunks** (int|numpy.array): 给定minibatch的Label块数。
        - **num_correct_chunks** （int|numpy.array）: 给定minibatch的Interface和Label的块数

.. _cn_api_fluid_merics_Auc:

Auc
>>>>

.. py:class:: class paddle.fluid.metrics.Auc(name, curve='ROC', num_thresholds=4095)

Auc度量适用于二分类。参考https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 。需要注意auc度量本身是用Python计算值。如果关心速度，请用fluid.layers.auc。

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






