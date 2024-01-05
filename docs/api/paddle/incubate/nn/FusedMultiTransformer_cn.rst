.. _cn_api_paddle_incubate_nn_FusedMultiTransformer:

FusedMultiTransformer
-------------------------------

.. py:function:: class paddle.incubate.nn.FusedMultiTransformer(embed_dim, num_heads, dim_feedforward, dropout_rate=0.0, activation='gelu', normalize_before=True, ln_scale_attrs=None, ln_bias_attrs=None, qkv_weight_attrs=None, qkv_bias_attrs=None, linear_weight_attrs=None, linear_bias_attrs=None, ffn_ln_scale_attrs=None, ffn_ln_bias_attrs=None, ffn1_weight_attrs=None, ffn1_bias_attrs=None, ffn2_weight_attrs=None, ffn2_bias_attrs=None, epsilon=1e-05, num_layers=- 1, nranks=1, trans_qkvw=True, ring_id=- 1, name=None)

FusedMultiTransformer 由多层变压器层组成，该层包含两个子层，即自（多头）注意力和前馈网络。

Transformer 层的功能与以下伪代码一致：

.. code-block:: text
    >>>
    >>> if pre_layer_norm:
    ...     out = layer_norm(x)
    ...     out = qkv_linear(out) + qkv_bias
    ... else:
    ...     out = qkv_linear(x) + qkv_bias
    >>> out = transpose(out, perm=[2, 0, 3, 1, 4])
    >>> # extract q, k and v from out.
    >>> q = out[0:1, ::]
    >>> k = out[1:2, ::]
    >>> v = out[2:3, ::]
    >>> out = q * k^t
    >>> out = attn_mask + out
    >>> out = softmax(out)
    >>> out = dropout(out)
    >>> out = out * v
    >>> out = transpose(out, perm=[0, 2, 1, 3])
    >>> out = linear(out)
    >>> if pre_layer_norm:
    ...     out = x + dropout(out + bias)
    ... else:
    ...     out = layer_norm(x + dropout(out + bias))

    >>> residual = out;
    >>> if pre_layer_norm:
    ...     out = ffn_layer_norm(out)
    >>> out = ffn1_linear(out)
    >>> out = dropout(activation(out + ffn1_bias))
    >>> out = ffn2_linear(out)
    >>> out = residual + dropout(out + ffn2_bias)
    >>> if not pre_layer_norm:
    ...     out = ffn_layer_norm(out)

参数
::::::::::::
    - **embed_dim** (int) - 输入和输出中的预期特征尺寸。
    - **num_heads** (int) - 多头注意（MHA）的头数。
    - **dim_feedforward** (int) - 前馈网络（FFN）中的隐藏层大小。
    - **dropout_rate** (float, 可选) - 在 MHA 子层和 FFN 子层的前处理和后处理中使用了丢包率。默认值 0.0
    - **activation** (str,可选) - 前馈网络中的激活函数。默认的"gelu"
    - **normalize_before** (bool, 可选) - 指示是否将层归一化放入 MHA 和 FFN 子层的预处理中。如果为 True，则前处理为层归一化，后处理包括丢弃、剩余连接。另外，没有前处理和后处理，包括丢失，剩余连接，层归一化。默认值为真
    - **ln_scale_attrs** (ParamAttr|list|tuple, 可选) - 为注意力层_norm 指定权重参数属性。对于 Attention layer_norm 权重，如果它是一个列表/元组，[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。默认值：无，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ln_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 为"注意力"层_norm 指定偏倚参数属性。对于注意层_norm 偏见，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：无，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **qkv_weight_attrs** (ParamAttr|list|tuple,可选) - 为注意力 qkv 计算指定权重参数属性。注意 qkv 权重，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。默认值：无，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **qkv_bias_attrs** (ParamAttr|list|tuple|bool, 可选) - 指定"关注度"qkv 计算的偏倚参数属性。注意 qkv 的偏见，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的 a tr，而 attrs[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：无，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **linear_weight_attrs** (ParamAttr|list|tuple, 可选) - 指定"注意力"线性的权重参数属性。注意线性权重，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。默认值：无，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **linear_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 为"注意力"线性计算指定偏倚参数属性。对于注意力线性偏差，如果它是一个列表/元组，[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：无，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn_ln_scale_attrs** (ParamAttr|list|tuple, 可选) - 为 FFN layer_norm 指定权重参数属性。对于 FFN layer_norm 权重，如果它是一个列表/元组，[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。默认值：无，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn_ln_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 指定 FFN layer_norm 的偏置参数属性。对于 FFN layer_norm 偏差，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：无，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn1_weight_attrs** (ParamAttr|list|tuple, 可选) - 要指定 FFN 第一个线性的权重参数属性。对于 FFN 的第一线性权重，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。默认值：无，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn1_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 指定 FFN 第一个线性的偏倚参数属性。对于 FFN 的第一个线性偏差，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：无，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn2_weight_attrs** (ParamAttr|list|tuple, 可选) - 指定 FFN 第二线性的权重参数属性。对于 FFN 的第二线性权重，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。默认值：无，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn2_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 指定 FFN 第二线性的偏置参数属性。对于 FFN 第二线性偏置，如果它是一个列表/元组，attrs[0]将被用作变压器层 0 的属性，而属性[1]将被用作变压器层 1 的属性，等等。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：无，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **epsilon** (float, 可选) - 小浮点值被添加到 layer_norm 的分母以避免被零除。默认值：1e-05
    - **num_layers** (int, 可选) - 变压器的层数。如果 qkv_weight_attrs 是一个列表或元组，则从 qkv_weight_attrs 中获得层数。num_layers 仅在 qkv_weight_attrs 不是列表或元组时生效。默认值：—1
    - **nranks** (int, 可选) - 分布张量模型并行 nranks。默认为 1，表示不使用 mp。
    - **trans_qkvw** (bool, 可选) - 是否对 qkv 的权重进行转置。如果为真，qkv 的形状八个应该是【3,num_head,dim_head,tim_embed】。否则 qkv 的权值的形状应该是【dim_embed,3,num_head,dim_head】。默认值：真。
    - **ring_id** (int, 可选) - 用于分布式张量模型并行化。默认值为-1，表示不使用 mp。
    - **name** (str，可选) - 默认值为“无”。通常用户不需要设置此属性。如需详细资讯，请参阅:ref:`api_guide_Name`。

代码示例
:::::::::

COPY-FROM: paddle.incubate.nn.FusedMultiTransformer

.. py:function:: forward(src, attn_mask=None, caches=None, pre_caches=None, rotary_embs=None, rotary_emb_dims=0, seq_lens=None, time_step=None)

在输入上应用多个变压器层。

参数
::::::::::::
    - **src** (Tensor) - Transformer 层的输入。它是一个形状为【batch_size,sequence_length,d_model】的张量。数据类型应该是 float16 或 float32。
    - **attn_mask** (Tensor, 可选) - 在多头注意中使用的张量可以防止注意到一些不需要的位置，通常是填充位置或后续位置。它是一个形状较大的张量【批大小，1，序列长度，序列长度】。当没有什么想要或需要被阻止注意时，它可以是“无”。默认值为无。
    - **caches** (list(Tensor)|tuple(Tensor), 可选) - 推理生成模型的缓存结构张量。它仅用于推理，在训练时应设置为 None。形状为【2,batch_size,num_head,max_seq_len,head_dim】。默认值为无。
    - **pre_caches** (list(Tensor)|tuple(Tensor), 可选) - 生成模型的前缀缓存。形状是【2，bsz，num head，cache len，head dim】。默认值为无。
    - **rotary_embs** (Tensor 可选) - 为旋转计算嵌入了 RoPE。形状是【2，bsz，1，seq_len，head_dim】。默认值为无。
    - **rotary_emb_dims** (int, 可选) - 当 rotary_emb_dims 为 None 时，它为 0；当 rotary_embs 不为 None 且 pos_extra_ids 为 None 时，它为 1；当 rotary_embs 和 pos_extra_ids 都不为 None 时，它为 2。默认值为 0。
    - **seq_lens** (Tensor 可选) - 此批的序列长度。形状是【bsz】。默认值为无。
    - **time_step** (Tensor, 可选) - 生成模型的时间步长张量。用于解码阶段，用来表示时间步长，即 CacheKV 的实际序列。形状是[1]，必须在 CPUPlace 中。默认值为无。

返回
::::::::::::

如果缓存为 None，则返回一个与 src 具有相同形状和数据类型的张量，表示 Transformer 层的输出。如果 caches 不是 None，返回元组（output，caches），它的输出是 Transformer 层的输出，caches 与输入 caches 一起到位。

返回类型
::::::::::::
    - Tensor|tuple：如果 ``cache_kvs`` 为 None，则返回与 ``x`` 形状和数据类型相同的张量，代表变压器层的输出。如果 ``cache_kvs`` 不为 None，则返回元组（output, cache_kvs），其中 output 是变压器层的输出，cache_kvs 与输入`cache_kvs`原地更新。
