.. _cn_api_paddle_incubate_nn_FusedMultiHeadAttention:

FusedMultiHeadAttention
-------------------------------

.. py:class:: paddle.incubate.nn.FusedMultiHeadAttention(embed_dim, num_heads, dropout_rate=0.5, attn_dropout_rate=0.5, kdim=None, vdim=None, normalize_before=False, need_weights=False, qkv_weight_attr=None, qkv_bias_attr=None, linear_weight_attr=None, linear_bias_attr=None, pre_ln_scale_attr=None, pre_ln_bias_attr=None, ln_scale_attr=None, ln_bias_attr=None, epsilon=1e-5, nranks=1, ring_id=-1, transpose_qkv_wb=False, name=None)



**多头注意力机制**

注意力机制可以将查询（Query）与一组键值对（Key-Value）映射到输出。而多头注意力机制是将注意力机制的计算过程计算多次，以便模型提取不同子空间的信息。

细节可参考论文 `Attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 。

``FusedMultiHeadAttention`` 与已有的 ``MultiHeadAttention`` 有两处不同：

（1）表达的计算逻辑范围不同。相比 ``MultiHeadAttention`` ， ``FusedMultiHeadAttention`` 的前面在 ``normalize_before=True`` 时，多了 ``layer_norm`` 算子，后面多了 ``residual add`` ， ``dropout`` 和 ``layer_norm`` 的计算。

（2）q, k, v 的 weight 的存储格式不同。``MultiHeadAttention`` 将 q, k, v 的 weight 存储在三个 Tensor 中。``FusedMultiHeadAttention`` 的 q, k, v 的 weight 被统一存在一个权重 Tensor 中，其维度为 ``[3, num_heads, head_dim, embed_dim]`` ，当开启 ``transpose_qkv_wb`` 后，其维度为 ``[embed_dim, 3 * embed_dim]``。

参数
:::::::::
    - **embed_dim** (int) - 输入输出的维度。
    - **num_heads** (int) - 多头注意力机制的 Head 数量。
    - **dropout_rate** (float，可选) - Multi-Head Attention 后面的 dropout 算子的注意力目标的随机失活率。0 表示进行 dropout 计算。默认值：0.5。
    - **attn_dropout_rate** (float，可选) - Multi-Head Attention 中的 dropout 算子的注意力目标的随机失活率。0 表示不进行 dropout 计算。默认值：0.5。
    - **kdim** (int，可选) - 键值对中 key 的维度。如果为 ``None`` 则 ``kdim = embed_dim``。默认值 ``None`` 。
    - **vdim** (int，可选) - 键值对中 value 的维度。如果为 ``None`` 则 ``vdim = embed_dim``。默认值：``None`` 。
    - **normalize_before** (bool，可选) - 是 pre_layer_norm 结构（True）还是 post_layer_norm 结构（False）。pre_layer_norm 结构中，``layer_norm`` 算子位于 multi-head attention 和 ffn 的前面，post_layer_norm 结构中，``layer_norm`` 位于两者的后面。默认值：``False`` 。
    - **need_weights** (bool，可选) - 表明是否返回注意力权重。默认值：``False`` 。
    - **qkv_weight_attr** (ParamAttr，可选) - 为 Attention 中计算 q, k, v 时的计算指定权重参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **qkv_bias_attr** (ParamAttr，可选) - 为 Attention 中计算 q, k, v 时的计算指定偏置参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **linear_weight_attr** (ParamAttr，可选) - 为 Multi-Head Attention 之后的线性计算指定权重参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **linear_bias_attr** (ParamAttr，可选) - 为 Multi-Head Attention 之后的线性计算指定偏置参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **pre_ln_scale_attr** (ParamAttr，可选) - 为 pre_layer_norm 结构的前置 layer norm 计算指定权重参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **pre_ln_bias_attr** (ParamAttr，可选) - 为 pre_layer_norm 结构的前置 layer norm 计算指定偏置参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **ln_scale_attr** (ParamAttr，可选) - 为 post_layer_norm 结构的后置 layer norm 计算指定权重参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **ln_bias_attr** (ParamAttr，可选) - 为 post_layer_norm 结构的前置 layer norm 计算指定偏置参数属性的对象。默认值：``None``，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **epsilon** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **nranks** (int，可选) - 分布式 tensor parallel 运行下该策略的并行度。默认值：-1，表示不使用 tensor parallel。
    - **ring_id** (int，可选) - 分布式 tensor parallel 运行下通讯所使用的 NCCL id。默认值：-1，表示不使用 tensor parallel。
    - **transpose_qkv_wb** (bool，可选) - 生成 QKV 的映射计算的权重形状为 ``[3, num_heads, head_dim, embed_dim]`` (False) 还是 ``[embed_dim, 3 * embed_dim]`` (True)。仅支持 GPU 运算。默认值：``False``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **x** (Tensor)：默认形状为 ``[batch_size, sequence_length, embed_dim]``，其数据类型为 float32，float64 或者 float16。
    - **output** (Tensor)：其形状和数据类型与输入 x 相同。

返回
:::::::::
计算 FusedMultiHeadAttention 的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.incubate.nn.FusedMultiHeadAttention
