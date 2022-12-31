.. _cn_api_fluid_layers_crf_decoding:

crf_decoding
-------------------------------


.. py:function::  paddle.static.nn.crf_decoding(input, param_attr, label=None, length=None)



该层读取由 :ref:`cn_api_fluid_layers_linear_chain_crf` 学习的 emission feature weights（发射状态特征的权重）和 transition feature weights (转移特征的权重) 进行解码。
本层实现了 Viterbi 算法，可以动态地寻找隐藏状态最可能的序列，该序列也被称为 Viterbi 路径（Viterbi path），从而得到观察标签 (tags) 序列。

这个层运算的结果会随着输入 ``Label`` 的有无而改变：

      1. ``Label`` 非 None 的情况，在实际训练中时常发生。此时本层会协同 :ref:`cn_api_fluid_layers_chunk_eval` 工作。在 LoD 模式下，本层会返回一行形为 [N X 1]  的向量，在 padding 模式下，返回形状则为 [B x S]，其中值为 0 的部分代表该 label 不适合作为对应结点的标注，值为 1 的部分则反之。此类型的输出可以直接作为 :ref:`cn_api_fluid_layers_chunk_eval` 算子的输入；

      2. 当没有 ``Label`` 时，该函数会执行标准解码过程；

（没有 ``Label`` 时）该运算返回一个形状为 [N X 1] 或 [B x S] 的向量，此处的形状取决于输入是否为带有 LoD 信息的 Tensor，其中元素取值范围为 0 ~ 最大标注个数-1，分别为预测出的标注（tag）所在的索引。

参数
::::::::::::

    - **input** (Tensor) — 一个形为 [N x D] 的 Tensor，其中 N 是 mini-batch 的大小，D 是标注（tag) 的总数；或者形为 [B x S x D] 的普通 Tensor，B 是批次大小，S 是序列最大长度，D 是标注的总数。该输入是 :ref:`cn_api_fluid_layers_linear_chain_crf`` 的 unscaled emission weight matrix （未标准化的发射权重矩阵）。数据类型为 float32 或者 float64。
    - **param_attr** (ParamAttr，可选)：指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_guide_ParamAttr` 。
    - **label** (Tensor，可选) —  形为 [N x 1] 的正确标注（ground truth）（LoD 模式），或者形状为 [B x S]。有关该参数的更多信息，请详见上述描述。数据类型为 int64。
    - **length** (Tensor，可选) —  形状为 [B x 1]，表示输入序列的真实长度。该输入非 None，表示该层工作在 padding 模式下，即 ``input`` 和 ``label`` 都是带 padding 的普通 Tensor。数据类型为 int64。

返回
::::::::::::
Tensor，解码结果具体内容根据 ``Label`` 参数是否提供而定，请参照上面的介绍来详细了解。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.crf_decoding
