#################
 fluid.metrics
#################



.. _cn_api_fluid_metrics_Accuracy:

Accuracy
-------------------------------

.. py:class:: paddle.fluid.metrics.Accuracy(name=None)

计算多批次的平均准确率。
https://en.wikipedia.org/wiki/Accuracy_and_precision

参数:
    - **name** — 度量标准的名称

**代码示例**

.. code-block:: python

    # 假设有batch_size = 128
    batch_size=128
    accuracy_manager = fluid.metrics.Accuracy()
    # 假设第一个batch的准确率为0.9
    batch1_acc = 0.9
    accuracy_manager.update(value = batch1_acc, weight = batch_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" % (batch1_acc, accuracy_manager.eval()))
    # 假设第二个batch的准确率为0.8
    batch2_acc = 0.8
    accuracy_manager.update(value = batch2_acc, weight = batch_size)
    #batch1和batch2的联合准确率为(batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2
    print("expect accuracy: %.2f, get accuracy: %.2f" % ((batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2, accuracy_manager.eval()))
    #重置accuracy_manager
    accuracy_manager.reset()
    #假设第三个batch的准确率为0.8
    batch3_acc = 0.8
    accuracy_manager.update(value = batch3_acc, weight = batch_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" % (batch3_acc, accuracy_manager.eval()))



.. py:method:: update(value, weight)

更新mini batch的状态。

参数：    
    - **value** (float|numpy.array) – 每个mini batch的正确率
    - **weight** (int|float) – batch 大小


.. py:method:: eval()

返回所有累计batches的平均准确率（float或numpy.array）。


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











.. _cn_api_fluid_metrics_ChunkEvaluator:

ChunkEvaluator
-------------------------------

.. py:class:: paddle.fluid.metrics.ChunkEvaluator(name=None)

用mini-batch的chunk_eval累计counter numbers，用累积的counter numbers计算准确率、召回率和F1值。对于chunking的基础知识，请参考 .. _Chunking with Support Vector Machines: https://aclanthology.info/pdf/N/N01/N01-1025.pdf 。ChunkEvalEvaluator计算块检测（chunk detection）的准确率，召回率和F1值，支持IOB, IOE, IOBES和IO标注方案。

**代码示例**：

.. code-block:: python

        # 初始化chunck-level的评价管理。
        metric = fluid.metrics.ChunkEvaluator()
        
        # 假设模型预测10个chuncks，其中8个为正确，且真值有9个chuncks。
        num_infer_chunks = 10
        num_label_chunks = 9
        num_correct_chunks = 8
        
        metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
        numpy_precision, numpy_recall, numpy_f1 = metric.eval()
        
        print("precision: %.2f, recall: %.2f, f1: %.2f" % (numpy_precision, numpy_recall, numpy_f1))
         
        # 下一个batch，完美地预测了3个正确的chuncks。
        num_infer_chunks = 3
        num_label_chunks = 3
        num_correct_chunks = 3
         
        metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
        numpy_precision, numpy_recall, numpy_f1 = metric.eval()
         
        print("precision: %.2f, recall: %.2f, f1: %.2f" % (numpy_precision, numpy_recall, numpy_f1))
    
.. py:method:: update(num_infer_chunks, num_label_chunks, num_correct_chunks)

基于layers.chunk_eval()输出更新状态（state)输出

参数:
    - **num_infer_chunks** (int|numpy.array): 给定minibatch的Interface块数。
    - **num_label_chunks** (int|numpy.array): 给定minibatch的Label块数。
    - **num_correct_chunks** （int|float|numpy.array）: 给定minibatch的Interface和Label的块数







.. _cn_api_fluid_metrics_CompositeMetric:

CompositeMetric
-------------------------------

.. py:class:: paddle.fluid.metrics.CompositeMetric(name=None)

在一个实例中组合多个指标。例如，将F1、准确率、召回率合并为一个指标。

**代码示例**

.. code-block:: python

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








.. _cn_api_fluid_metrics_DetectionMAP:

DetectionMAP
-------------------------------

.. py:class:: paddle.fluid.metrics.DetectionMAP(input, gt_label, gt_box, gt_difficult=None, class_num=None, background_label=0, overlap_threshold=0.5, evaluate_difficult=True, ap_version='integral')

计算 detection 平均精度（mAP）。 mAP是衡量object detectors精度的指标，比如 Faster R-CNN,SSD等。它不同于召回率，它是最大精度的平均值。 5

通常步骤如下：

1. 根据detectors中的输入和label，计算  true positive 和 false positive
2. 计算map，支持 ‘11 point’ and ‘integral’

请从以下文章中获取更多信息：
    - https://sanchom.wordpress.com/tag/average-precision/
    - https://arxiv.org/abs/1512.0232

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

        import paddle.fluid.layers as layers
         
        batch_size = -1 # 可以为任意大小
        image_boxs_num = 10
        bounding_bboxes_num = 21
         
        pb = layers.data(name='prior_box', shape=[image_boxs_num, 4],
            append_batch_size=False, dtype='float32')
         
        pbv = layers.data(name='prior_box_var', shape=[image_boxs_num, 4],
            append_batch_size=False, dtype='float32')
         
        loc = layers.data(name='target_box', shape=[batch_size, bounding_bboxes_num, 4],
            append_batch_size=False, dtype='float32')
         
        scores = layers.data(name='scores', shape=[batch_size, bounding_bboxes_num, image_boxs_num],
            append_batch_size=False, dtype='float32')
         
        nmsed_outs = fluid.layers.detection_output(scores=scores,
            loc=loc, prior_box=pb, prior_box_var=pbv)
         
        gt_box = fluid.layers.data(name="gt_box", shape=[batch_size, 4], dtype="float32")
        gt_label = fluid.layers.data(name="gt_label", shape=[batch_size, 1], dtype="float32")
        difficult = fluid.layers.data(name="difficult", shape=[batch_size, 1], dtype="float32")
        
        exe = fluid.Executor(fluid.CUDAPlace(0))
        map_evaluator = fluid.metrics.DetectionMAP(nmsed_outs, gt_label, gt_box, difficult, class_num = 3)
        cur_map, accum_map = map_evaluator.get_map_var()

        # 更详细的例子请参见
        # https://github.com/PaddlePaddle/models/blob/43cdafbb97e52e6d93cc5bbdc6e7486f27665fc8/PaddleCV/object_detection



.. py:method:: get_map_var()

返回：当前 mini-batch 的 mAP 变量，和跨 mini-batch 的 mAP 累加和

.. py:method::  reset(executor, reset_program=None)

在指定 batch 的每一 pass/user  开始时重置度量状态。

参数：
    - **executor** (Executor) – 执行reset_program的执行程序
    - **reset_program** (Program|None) –  单一 program 的 reset 过程。如果设置为 None，将创建一个 program



.. _cn_api_fluid_metrics_EditDistance:

EditDistance
-------------------------------

.. py:class:: paddle.fluid.metrics.EditDistance(name)

编辑距离是通过计算将一个字符串转换为另一个字符串所需的最小编辑操作数（添加、删除或替换）来量化两个字符串（例如单词）彼此不相似的程度一种方法。
参考 https://en.wikipedia.org/wiki/Edit_distance。
此EditDistance类使用更新函数获取两个输入：
    1. distance：一个形状为（batch_size, 1）的numpy.array，每个元素表示两个序列之间的编辑距离；
    2. seq_num：一个整型/浮点型数，代表序列对的数目，并返回多个序列对的整体编辑距离。

参数:
    - **name** - 度量标准名称

**代码示例**

.. code-block:: python

    import numpy as np
    
    # 假设batch_size为128
    batch_size = 128
    
    # 初始化编辑距离管理器
    distances_evaluator = fluid.metrics.EditDistance("EditDistance")
    # 生成128个序列对间的编辑距离，此处的最大距离是10
    edit_distances_batch0 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
    seq_num_batch0 = batch_size

    distance_evaluator.update(edit_distances_batch0, seq_num_batch0)
    distance, instance_error = distance_evaluator.eval()
    avg_distance, wrong_instance_ratio = distance_evaluator.eval()
    print("the average edit distance for batch0 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))
    edit_distances_batch1 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
    seq_num_batch1 = batch_size

    distance_evaluator.update(edit_distances_batch1, seq_num_batch1)
    avg_distance, wrong_instance_ratio = distance_evaluator.eval()
    print("the average edit distance for batch0 and batch1 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))


.. py:method:: distance_evaluator.reset()

.. code-block:: python

  edit_distances_batch2 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
  seq_num_batch2 = batch_size
  distance_evaluator.update(edit_distances_batch2, seq_num_batch2)
  avg_distance, wrong_instance_ratio = distance_evaluator.eval()
  print("the average edit distance for batch2 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))


.. py:method:: update(distances, seq_num)

更新整体的编辑距离。

参数：
    - **distances** – 一个形状为(batch_size, 1)的numpy.array，每个元素代表两个序列间的距离。(edit) – 
    - **seq_num** – 一个整型/浮点型值，代表序列对的数量。


.. py:method:: eval()

返回两个浮点数：
avg_distance：使用更新函数更新的所有序列对的平均距离。
avg_instance_error：编辑距离不为零的序列对的比例。





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

Precision(也称为 positive predictive value,正预测值)是被预测为正样例中实际为正的比例。
https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
该类管理二分类任务的precision分数。



**代码示例**

.. code-block:: python

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







.. _cn_api_fluid_metrics_Recall:

Recall
-------------------------------

.. py:class:: paddle.fluid.metrics.Recall(name=None)

召回率（也称为敏感度）是指得到的相关实例数占相关实例总数的比重

https://en.wikipedia.org/wiki/Precision_and_recall

该类管理二分类任务的召回率。

**代码示例**

.. code-block:: python

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









