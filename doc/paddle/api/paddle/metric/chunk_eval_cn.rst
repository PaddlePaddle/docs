.. _cn_api_fluid_layers_chunk_eval:

chunk_eval
-------------------------------

.. py:function:: paddle.fluid.layers.chunk_eval(input, label, chunk_scheme, num_chunk_types, excluded_chunk_types=None, sqe_length=None)




该OP计算语块识别（chunk detection）的准确率、召回率和F1值，常用于命名实体识别（NER，语块识别的一种）等序列标注任务中。

语块识别的基础请参考 `Chunking with Support Vector Machines <https://www.aclweb.org/anthology/N01-1025>`_

该OP支持IOB，IOE，IOBES和IO（plain）的标注方式。以下是这些标注方式在命名实体识别示例中的使用：

::


    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
           Li     Ming    works  at  Agricultural   Bank   of    China  in  Beijing.
    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
    IO     I-PER  I-PER   O      O   I-ORG          I-ORG  I-ORG I-ORG  O   I-LOC
    IOB    B-PER  I-PER   O      O   B-ORG          I-ORG  I-ORG I-ORG  O   B-LOC
    IOE    I-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   E-LOC
    IOBES  B-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   S-LOC
    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========

例中有PER（人名），ORG（机构名）和LOC（地名）三种语块类型（命名实体类型）。可以看到，一个完整的标签包括标注类型（tag type）和语块类型（chunk type），形式为 ``标注类型-语块类型（tag type-chunk type）`` 。

由于该OP在计算实现上使用的是标签id而非标签字符串，为使其能正确运行，标签id要能够转换为相应的标注类型（tag type）和语块类型（chunk type）。该OP使用了下面的方式完成映射转换：

::


    tag_type = label % num_tag_type
    chunk_type = label / num_tag_type

其中num_tag_type是标注方式中的标签类型（tag type）数，各标注方式的tag type取值如下：

::


    Scheme Begin Inside End   Single
    plain   0     -      -     -
    IOB     0     1      -     -
    IOE     -     0      1     -
    IOBES   0     1      2     3

据此，在上面的NER例子中，若标注方式是IOB，语块类型包括ORG、PER和LOC三种，则所有标签及其对应id如下：

::


    B-ORG  0
    I-ORG  1
    B-PER  2
    I-PER  3
    B-LOC  4
    I-LOC  5
    O      6

从标签id可以正确的得到其对应的标注类型（tag type）和语块类型（chunk type）。

参数：
    - **input** (Variable) - 表示网络预测的标签，为Tensor或LoD level为1的LoDTensor。Tensor时，其形状为 :math:`[N, M, 1]` ，其中 :math:`N` 表示batch size， :math:`M` 表示序列长度；LoDTensor时，其形状为 :math:`[N, 1]` 或 :math:`[N]` ，其中 :math:`N` 表示所有序列长度之和。数据类型为int64。
    - **label** (Variable) - 表示真实标签（ground-truth）的Tensor或LoDTensor，和 ``input`` 具有相同形状、LoD和数据类型。
    - **chunk_scheme** (str) - 标注方式，必须是IOB，IOE，IOBES或者plain中的一种。
    - **num_chunk_types** (int) - 表示标签中的语块类型数。
    - **excluded_chunk_types** (list，可选) - 表示不计入统计的语块类型，需要为语块类型（int表示）的列表。默认值为空的list。
    - **seq_length** (Variable，可选) - 当输入 ``input`` 和 ``label`` 是Tensor而非LoDTensor时，用来指示输入中每个序列长度的1-D Tensor。数据类型为int64。可以为空，默认为None。

返回：Variable的元组。元组中包含准确率、召回率、F1值，以及识别出的语块数目、标签中的语块数目、正确识别的语块数目。每个均是单个元素的Tensor，准确率、召回率、F1值的数据类型为float32，其他的数据类型为int64。

返回类型：tuple

**代码示例**：

.. code-block:: python:

    import paddle.fluid as fluid
     
    dict_size = 10000
    label_dict_len = 7
    sequence = fluid.data(
        name='id', shape=[None, 1], lod_level=1, dtype='int64')
    embedding = fluid.embedding(
        input=sequence, size=[dict_size, 512])
    hidden = fluid.layers.fc(input=embedding, size=512)
    label = fluid.data(
        name='label', shape=[None, 1], lod_level=1, dtype='int64')
    crf = fluid.layers.linear_chain_crf(
        input=hidden, label=label, param_attr=fluid.ParamAttr(name="crfw"))
    crf_decode = fluid.layers.crf_decoding(
        input=hidden, param_attr=fluid.ParamAttr(name="crfw"))
    fluid.layers.chunk_eval(
        input=crf_decode,
        label=label,
        chunk_scheme="IOB",
        num_chunk_types=int((label_dict_len - 1) / 2))









