.. _cn_api_fluid_layers_chunk_eval:

chunk_eval
-------------------------------

.. py:function:: paddle.fluid.layers.chunk_eval(input, label, chunk_scheme, num_chunk_types, excluded_chunk_types=None)

块估计（Chunk Evaluator）

该功能计算并输出块检测（chunk detection）的准确率、召回率和F1值。

chunking的一些基础请参考 `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_

ChunkEvalOp计算块检测（chunk detection）的准确率、召回率和F1值，并支持IOB，IOE，IOBES和IO标注方案。以下是这些标注方案的命名实体（NER）标注例子：

::


    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
           Li     Ming    works  at  Agricultural   Bank   of    China  in  Beijing.
    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
    IO     I-PER  I-PER   O      O   I-ORG          I-ORG  I-ORG I-ORG  O   I-LOC
    IOB    B-PER  I-PER   O      O   B-ORG          I-ORG  I-ORG I-ORG  O   B-LOC
    IOE    I-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   E-LOC
    IOBES  B-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   S-LOC
    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========

有三种块类别（命名实体类型），包括PER（人名），ORG（机构名）和LOC（地名），标签形式为标注类型（tag type）-块类型（chunk type）。

由于计算实际上用的是标签id而不是标签，需要额外注意将标签映射到相应的id，这样CheckEvalOp才可运行。关键在于id必须在列出的等式中有效。

::


    tag_type = label % num_tag_type
    chunk_type = label / num_tag_type

num_tag_type是标注规则中的标签类型数，num_chunk_type是块类型数，tag_type从下面的表格中获取值。

::


    Scheme Begin Inside End   Single
    plain   0     -      -     -
    IOB     0     1      -     -
    IOE     -     0      1     -
    IOBES   0     1      2     3

仍以NER为例，假设标注规则是IOB块类型为ORG，PER和LOC。为了满足以上等式，标签图如下：

::


    B-ORG  0
    I-ORG  1
    B-PER  2
    I-PER  3
    B-LOC  4
    I-LOC  5
    O      6

不难证明等式的块类型数为3，IOB规则中的标签类型数为2.例如I-LOC的标签id为5，I-LOC的标签类型id为1，I-LOC的块类型id为2，与等式的结果一致。

参数：
    - **input** (Variable) - 网络的输出预测
    - **label** (Variable) - 测试数据集的标签
    - **chunk_scheme** (str) - 标注规则，表示如何解码块。必须数IOB，IOE，IOBES或者plain。详情见描述
    - **num_chunk_types** (int) - 块类型数。详情见描述
    - **excluded_chunk_types** (list) - 列表包含块类型id，表示不在计数内的块类型。详情见描述

返回：元组（tuple），包含precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks

返回类型：tuple（元组）

**代码示例**：

.. code-block:: python:

    import paddle.fluid as fluid
     
    dict_size = 10000
    label_dict_len = 7
    sequence = fluid.layers.data(
        name='id', shape=[1], lod_level=1, dtype='int64')
    embedding = fluid.layers.embedding(
        input=sequence, size=[dict_size, 512])
    hidden = fluid.layers.fc(input=embedding, size=512)
    label = fluid.layers.data(
        name='label', shape=[1], lod_level=1, dtype='int32')
    crf = fluid.layers.linear_chain_crf(
        input=hidden, label=label, param_attr=fluid.ParamAttr(name="crfw"))
    crf_decode = fluid.layers.crf_decoding(
        input=hidden, param_attr=fluid.ParamAttr(name="crfw"))
    fluid.layers.chunk_eval(
        input=crf_decode,
        label=label,
        chunk_scheme="IOB",
        num_chunk_types=(label_dict_len - 1) / 2)









