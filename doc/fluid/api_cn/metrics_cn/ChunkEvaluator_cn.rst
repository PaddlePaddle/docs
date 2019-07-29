.. _cn_api_fluid_metrics_ChunkEvaluator:

ChunkEvaluator
-------------------------------

.. py:class:: paddle.fluid.metrics.ChunkEvaluator(name=None)

用mini-batch的chunk_eval累计counter numbers，用累积的counter numbers计算准确率、召回率和F1值。对于chunking的基础知识，请参考 .. _Chunking with Support Vector Machines: https://aclanthology.info/pdf/N/N01/N01-1025.pdf 。ChunkEvalEvaluator计算块检测（chunk detection）的准确率，召回率和F1值，支持IOB, IOE, IOBES和IO标注方案。

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

基于layers.chunk_eval()输出更新状态（state)输出

参数:
    - **num_infer_chunks** (int|numpy.array): 给定minibatch的Interface块数。
    - **num_label_chunks** (int|numpy.array): 给定minibatch的Label块数。
    - **num_correct_chunks** （int|float|numpy.array）: 给定minibatch的Interface和Label的块数







