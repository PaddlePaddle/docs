.. _cn_api_fluid_metrics_ChunkEvaluator:

ChunkEvaluator
-------------------------------

.. py:class:: paddle.fluid.metrics.ChunkEvaluator(name=None)
该接口使用mini-batch的chunk_eval累计的counter numbers，来计算准确率、召回率和F1值。ChunkEvaluator有三个状态num_infer_chunks，num_label_chunks和num_correct_chunks，分别对应语块数目、标签中的语块数目、正确识别的语块数目。对于chunking的基础知识，请参考 https://www.aclweb.org/anthology/N01-1025 。ChunkEvalEvaluator计算块检测（chunk detection）的准确率，召回率和F1值，支持IOB, IOE, IOBES和IO标注方案。

参数：
    - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：初始化后的 ``ChunkEvaluator`` 对象

返回类型：ChunkEvaluator

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid

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

该函数使用输入的(num_infer_chunks, num_label_chunks, num_correct_chunks)来累计更新ChunkEvaluator对象的对应状态，更新方式如下：
    
    .. math:: 
                   \\ \begin{array}{l}{\text { self. num_infer_chunks }+=\text { num_infer_chunks }} \\ {\text { self. num_Label_chunks }+=\text { num_label_chunks }} \\ {\text { self. num_correct_chunks }+=\text { num_correct_chunks }}\end{array} \\

参数:
    - **num_infer_chunks** (int|numpy.array) – 给定mini-batch的语块数目。
    - **num_label_chunks** (int|numpy.array) - 给定mini-batch的标签中的语块数目。
    - **num_correct_chunks** （int|numpy.array）— 给定mini-batch的正确识别的语块数目。

返回：无

.. py:method:: eval()

该函数计算并返回准确率，召回率和F1值。

返回：准确率，召回率和F1值

返回类型：float

