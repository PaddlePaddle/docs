.. _cn_api_paddle_incubate_nn_FusedMultiTransformer:

FusedMultiTransformer
-------------------------------

.. py:class:: paddle.incubate.nn.FusedMultiTransformer(embed_dim, num_heads, dim_feedforward, dropout_rate=0.0, activation='gelu', normalize_before=True, ln_scale_attrs=None, ln_bias_attrs=None, qkv_weight_attrs=None, qkv_bias_attrs=None, linear_weight_attrs=None, linear_bias_attrs=None, ffn_ln_scale_attrs=None, ffn_ln_bias_attrs=None, ffn1_weight_attrs=None, ffn1_bias_attrs=None, ffn2_weight_attrs=None, ffn2_bias_attrs=None, epsilon=1e-05, num_layers=- 1, nranks=1, trans_qkvw=True, ring_id=- 1, name=None)

FusedMultiTransformer 由多层 Transformer 层组成，该层包含两个子层，即自（多头）注意力和前馈网络。

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
    - **embed_dim** (int) - 输入和输出中的预期特征大小。
    - **num_heads** (int) - 多头注意力 （MHA） 中的头部数量。
    - **dim_feedforward** (int) - 前馈网络（FFN）中的隐藏层大小。
    - **dropout_rate** (float, 可选) - MHA 和 FFN 子层的预处理和 post-precess 中使用的丢失概率。默认值：0.0
    - **activation** (str,可选) - 前馈网络中的激活函数。默认为"gelu"
    - **normalize_before** (bool, 可选) - 指示是否将层归一化放入 MHA 和 FFN 子层的预处理中。如果为 True，则预处理是层归一化，后处理包括丢弃、残差连接。否则，没有预处理和 post-precess，包括丢失、残差连接、层归一化。默认值：True
    - **ln_scale_attrs** (ParamAttr|list|tuple, 可选) - 指定 Attention layer_norm 的权重参数属性。 对于 Attention layer_norm 权重，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1] 将用作 transformer 第 1 层的 attr，依此类推。否则，所有图层都将其用作创建参数的属性。默认值：None，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ln_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 指定 Attention layer_norm 的偏置参数属性。对于 Attention layer_norm 偏差，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1] 将用作 transformer 第 1 层的 attr，依此类推。否则，所有图层都将其用作创建参数的属性。False 值表示相应的层没有可训练的偏差参数。默认值：None，表示使用默认偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **qkv_weight_attrs** (ParamAttr|list|tuple,可选) - 指定 Attention qkv 计算的权重参数属性。对于 Attention qkv 权重，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1] 将用作 transformer 第 1 层的 attr，依此类推。否则，所有图层都将其用作创建参数的属性。默认值：None，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **qkv_bias_attrs** (ParamAttr|list|tuple|bool, 可选) - 指定 Attention qkv 计算的偏置参数属性。对于 Attention qkv 偏差，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1] 将用作 transformer 第 1 层的 attr，依此类推。否则，所有图层都将其用作创建参数的属性。False 值表示相应的层没有可训练的偏差参数。默认值：None，表示使用默认偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **linear_weight_attrs** (ParamAttr|list|tuple, 可选) - 指定 Attention linear 的权重参数属性。对于 Attention linear 权重，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1] 将用作 transformer 第 1 层的 attr，依此类推。否则，所有图层都将其用作创建参数的属性。默认值：None，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **linear_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 指定 Attention linear 的偏置参数属性。对于 Attention linear 偏差，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1]将被用作 transformer 第 1 层的 attr，依此类推。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：None，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn_ln_scale_attrs** (ParamAttr|list|tuple, 可选) - 指定 FFN layer_norm 的权重参数属性。对于 FFN layer_norm 权重，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1]将被用作 transformer 第 1 层的 attr，依此类推。否则，所有层都使用它作为 attr 来创建参数。默认值：None，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn_ln_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 指定 FFN layer_norm 的偏置参数属性。对于 FFN layer_norm 偏差，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1]将被用作 transformer 第 1 层的 attr，依此类推。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：None，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn1_weight_attrs** (ParamAttr|list|tuple, 可选) - 指定 FFN first linear 的权重参数属性。对于 FFN first linear 权重，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1]将被用作 transformer 第 1 层的 attr，依此类推。否则，所有层都使用它作为 attr 来创建参数。默认值：无，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn1_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 指定 FFN first linear 的偏置参数属性。对于 FFN first linear 偏差，如果它是一个列表/元组，则 attrs[0] 将用作 transformer 层 0 的 attr，attrs[1]将被用作 transformer 第 1 层的 attr，依此类推。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：None，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn2_weight_attrs** (ParamAttr|list|tuple, 可选) - 指定 FFN second linear 的权重参数属性。对于 FFN second linear 权重，如果它是一个列表/元组，attrs[0] 将用作 transformer 层 0 的 attr，attrs[1]将被用作 transformer 第 1 层的 attr，依此类推。否则，所有层都使用它作为 attr 来创建参数。默认值：无，表示使用默认权重参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **ffn2_bias_attrs** (ParamAttr|list|tuple|bool,可选) - 指定 FFN second linear 的偏置参数属性。对于 FFN second linear 偏置，如果它是一个列表/元组，attrs[0] 将用作 transformer 层 0 的 attr，attrs[1]将被用作 transformer 第 1 层的 attr，依此类推。否则，所有层都使用它作为 attr 来创建参数。False 值意味着相应的层没有可训练的偏差参数。默认值：None，这意味着使用默认的偏置参数属性。有关详细信息，请参阅 ParamAttr 中的用法。
    - **epsilon** (float, 可选) - 将小浮点值添加到 layer_norm 的分母上，以避免除以零。默认值：1e-05。
    - **num_layers** (int, 可选) - transformer 的层数。如果 qkv_weight_attrs 是列表或元组，则从 qkv_weight_attrs 中获取层数。仅当 qkv_weight_attrs 不是列表或元组时，num_layers 才会生效。默认值：-1。
    - **nranks** (int, 可选) - 分布式张量模型并行 nranks。默认值为 1，表示不使用 mp。
    - **trans_qkvw** (bool, 可选) - 是否转置 qkv 的权重。如果为 true，则 qkv 的形状八应为 [3， num_head， dim_head， dim_embed]。否则，qkv 的权重形状应为 [dim_embed， 3， num_head， dim_head]。默认值：True。
    - **ring_id** (int, 可选) - 用于分布式张量模型并行。默认值为-1，表示不使用 mp。
    - **name** (str，可选) - 默认值为 None。通常，用户不需要设置此属性。有关更多信息，请参阅:ref:`api_guide_Name`。

代码示例
:::::::::

COPY-FROM: paddle.incubate.nn.FusedMultiTransformer
