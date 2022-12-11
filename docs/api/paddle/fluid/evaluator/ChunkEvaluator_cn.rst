.. _cn_api_fluid_metrics_ChunkEvaluator:

ChunkEvaluator
-------------------------------

.. py:class:: paddle.fluid.metrics.ChunkEvaluator(name=None)



该接口使用mini-batch的chunk_eval累计的counter numbers，来计算准确率、召回率和F1值。ChunkEvaluator有三个状态num_infer_chunks，num_label_chunks和num_correct_chunks，分别对应语块数目、标签中的语块数目、正确识别的语块数目。对于chunking的基础知识，请参考 https://www.aclweb.org/anthology/N01-1025 。ChunkEvalEvaluator计算块检测（chunk detection）的准确率，召回率和F1值，支持IOB, IOE, IOBES和IO标注方案。

参数
::::::::::::

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
初始化后的 ``ChunkEvaluator`` 对象

返回类型
::::::::::::
ChunkEvaluator

代码示例
::::::::::::


COPY-FROM: paddle.fluid.metrics.ChunkEvaluator

方法
::::::::::::
update(num_infer_chunks, num_label_chunks, num_correct_chunks)
'''''''''

该函数使用输入的(num_infer_chunks, num_label_chunks, num_correct_chunks)来累计更新ChunkEvaluator对象的对应状态，更新方式如下：
    
    .. math:: 
                   \\ \begin{array}{l}{\text { self. num_infer_chunks }+=\text { num_infer_chunks }} \\ {\text { self. num_Label_chunks }+=\text { num_label_chunks }} \\ {\text { self. num_correct_chunks }+=\text { num_correct_chunks }}\end{array} \\

**参数**

    - **num_infer_chunks** (int|numpy.array) – 给定mini-batch的语块数目。
    - **num_label_chunks** (int|numpy.array) - 给定mini-batch的标签中的语块数目。
    - **num_correct_chunks** （int|numpy.array）— 给定mini-batch的正确识别的语块数目。

**返回**
无

eval()
'''''''''

该函数计算并返回准确率，召回率和F1值。

**返回**
准确率，召回率和F1值

**返回类型**
float

