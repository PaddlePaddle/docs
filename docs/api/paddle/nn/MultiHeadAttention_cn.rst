.. _cn_api_paddle_nn_MultiHeadAttention:

MultiHeadAttention
-------------------------------

.. py:class:: paddle.nn.MultiHeadAttention(embed_dim, num_heads, dropout=0.0, kdim=None, vdim=None, need_weights=False, weight_attr=None, bias_attr=None)



**多头注意力机制**

注意力机制可以将查询（Query）与一组键值对（Key-Value）映射到输出。而多头注意力机制是将注意力机制的计算过程计算多次，以便模型提取不同子空间的信息。

细节可参考论文 `Attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 。


参数
::::::::::::

    - **embed_dim** (int) - 输入输出的维度。
    - **num_heads** (int) - 多头注意力机制的 Head 数量。
    - **dropout** (float，可选) - 注意力目标的随机失活率。0 表示不加 dropout。默认值：0。
    - **kdim** (int，可选) - 键值对中 key 的维度。如果为 ``None`` 则 ``kdim = embed_dim``。默认值：``None``。
    - **vdim** (int，可选) - 键值对中 value 的维度。如果为 ``None`` 则 ``kdim = embed_dim``。默认值：``None``。
    - **need_weights** (bool，可选) - 表明是否返回注意力权重。默认值：``False``。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **bias_attr** （ParamAttr，可选）- 指定偏置参数属性的对象。默认值：``None``，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。


代码示例
::::::::::::

COPY-FROM: paddle.nn.MultiHeadAttention
