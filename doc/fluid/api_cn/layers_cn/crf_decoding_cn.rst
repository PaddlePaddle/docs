.. _cn_api_fluid_layers_crf_decoding:

crf_decoding
-------------------------------

.. py:function::  paddle.fluid.layers.crf_decoding(input, param_attr, label=None)

该函数读取由 ``linear_chain_crf`` 学习的emission feature weights（发射状态特征的权重）和 transition feature weights(转移特征的权重)。
本函数实现了Viterbi算法，可以动态地寻找隐藏状态最可能的序列，该序列也被称为Viterbi路径（Viterbi path），从而得出的标注(tags)序列。

这个运算的结果会随着 ``Label`` 参数的有无而改变：

      1. ``Label`` 非None的情况，在实际训练中时常发生。此时本函数会协同 ``chunk_eval`` 工作。本函数会返回一行形为[N X 1]的向量，其中值为0的部分代表该label不适合作为对应结点的标注，值为1的部分则反之。此类型的输出可以直接作为 ``chunk_eval`` 算子的输入

      2. 当没有 ``Label`` 时，该函数会执行标准decoding过程

（没有 ``Label`` 时）该运算返回一个形为 [N X 1]的向量，其中元素取值范围为 0 ~ 最大标注个数-1，分别为预测出的标注（tag）所在的索引。

参数：
    - **input** (Variable)(LoDTensor，默认类型为 LoDTensor<float>) — 一个形为 [N x D] 的LoDTensor，其中 N 是mini-batch的大小，D是标注（tag) 的总数。 该输入是 ``linear_chain_crf`` 的 unscaled emission weight matrix （未标准化的发射权重矩阵）
    - **param_attr** (ParamAttr) — 参与训练的参数的属性
    - **label** (Variable)(LoDTensor，默认类型为 LoDTensor<int64_t>) —  形为[N x 1]的正确标注（ground truth）。 该项可选择传入。 有关该参数的更多信息，请详见上述描述

返回：(LoDTensor, LoDTensor<int64_t>)decoding结果。具体内容根据 ``Label`` 参数是否提供而定。请参照函数介绍来详细了解。

返回类型： Variable


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    images = fluid.layers.data(name='pixel', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int32')
    hidden = fluid.layers.fc(input=images, size=2)
    crf = fluid.layers.linear_chain_crf(input=hidden, label=label,
            param_attr=fluid.ParamAttr(name="crfw"))
    crf_decode = fluid.layers.crf_decoding(input=hidden,
            param_attr=fluid.ParamAttr(name="crfw"))












