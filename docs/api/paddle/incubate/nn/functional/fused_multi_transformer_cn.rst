.. _cn_api_paddle_incubate_nn_functional_fused_multi_transformer:

fused_multi_transformer
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_multi_transformer(x, ln_scales, ln_biases, qkv_weights, qkv_biases, linear_weights, linear_biases, ffn_ln_scales, ffn_ln_biases, ffn1_weights, ffn1_biases, ffn2_weights, ffn2_biases, pre_layer_norm=True, epsilon=1e-05, cache_kvs=None, pre_caches=None, seq_lens=None, rotary_embs=None, time_step=None, attn_mask=None, dropout_rate=0.0, rotary_emb_dims=0, activation='gelu', training=False, mode='upscale_in_train', trans_qkvw=True, ring_id=- 1, name=None)

这是一个融合算子，用于计算 Transformer 模型架构中的多个 transformer 层。

fused_multi_transformer 算子仅支持在 GPU 上运行。

Transformer 层的功能与以下伪代码一致：

.. code-block:: text

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
    - **x** (Tensor) - 输入张量可以是 3-D 张量，输入数据类型可以是 float16 或 float32，形状为`[batch\_size, sequence\_length, d\_model]`。
    - **ln_scales** (list(Tensor)|tuple(Tensor)) - 注意力机制中层归一化层的权重张量，形状为`[d\_model]`。
    - **ln_biases** (list(Tensor)|tuple(Tensor)) - 注意力机制中层归一化层的偏重张量，形状为`[d\_model]`。
    - **qkv_weights** (list(Tensor)|tuple(Tensor)) - 注意力 qkv 计算的权重张量，形状为`[3, num\_head, dim\_head, d\_model]`。
    - **qkv_biases** (list(Tensor)|tuple(Tensor)|None) - 注意力 qkv 计算的偏置张量，形状为`[3, num\_head, dim\_head]`。
    - **linear_weights** (list(Tensor)|tuple(Tensor)) - 注意力机制中线性层的权重张量，形状为`[num\_head * dim\_head, d\_model]`。
    - **linear_biases** (list(Tensor)|tuple(Tensor)|None) - 注意力机制中线性层的的偏置张量，形状为`[d\_model]`。
    - **ffn_ln_scales** (list(Tensor)|tuple(Tensor)) - 前馈层中层归一化层的权重张量，形状为`[d\_model]`。
    - **ffn_ln_biases** (list(Tensor)|tuple(Tensor)) - 前馈层中层归一化层的偏置张量，形状为`[d\_model]`。
    - **ffn1_weights** (list(Tensor)|tuple(Tensor)) - 前馈层中第一个线性变换层的权重张量，形状为`[d\_model, dim\_feedforward]`。
    - **ffn1_biases** (list(Tensor)|tuple(Tensor)|None) - 前馈层中第一个线性变换层的偏置张量，形状为`[dim\_feedforward]`。
    - **ffn2_weights** (list(Tensor)|tuple(Tensor)) - 前馈层中第二线性变换层的权重张量，形状为`[dim\_feedforward, d\_model]`。
    - **ffn2_biases** (list(Tensor)|tuple(Tensor)|None) - 前馈层中第二线性变换层的偏置张量，形状为`[d_model]`。
    - **pre_layer_norm** (bool，可选) - 是否是 pre_layer_norm（True）或 post_layer_norm（False）。默认为 True。
    - **epsilon** (float，可选) - 添加到 layer_norm 的分母中的小浮点值，以避免除以零。默认为 1e-5。
    - **cache_kvs** (list(Tensor)|tuple(Tensor)，可选) - 生成模型的缓存结构张量。形状为`[2, bsz, num\_head, max\_seq\_len, head\_dim]`。默认为 None。
    - **pre_caches** (list(Tensor)|tuple(Tensor)，可选) - 生成模型的前缀缓存。形状为`[2, bsz, num\_head, cache\_len, head\_dim]`。默认为 None。
    - **seq_lens** (Tensor，可选) - 此批次的序列长度。形状为`[bsz]`。默认为 None。
    - **rotary_embs** (Tensor，可选) - 用于旋转计算的 RoPE 嵌入。形状为`[2, bsz, 1, seq\_len, head\_dim]`。默认为 None。
    - **time_step** (Tensor，可选) - 生成模型的时间步张量。用于解码阶段，表示时间步，即 CacheKV 的实际 seq_len。形状为`[1]`，必须位于 CPUPlace。默认为 None。
    - **attn_mask** (Tensor，可选) - 用于多头注意力层中防止对某些不需要的位置（通常是填充或后续位置）进行注意。其形状为`[batch_size, 1, sequence_length, sequence_length]`。默认为 None。
    - **dropout_rate** (float，可选) - 将单元设置为零的 dropout 概率。默认为 0.0。
    - **rotary_emb_dims** (int，可选) - 旋转计算的 rotary_emb_dims，当 rotary_embs 为 None 时为 0，当 rotary_embs 不为 None 且 pos_extra_ids 为 None 时为 1，当 rotary_embs 和 pos_extra_ids 均不为 None 时为 2。默认为 0。
    - **activation** (str，可选) - 激活函数。默认为"gelu"。
    - **training** (bool，可选) - 标志是否处于训练阶段。默认为 False。
    - **mode** (str，可选) - ['upscale_in_train'(默认) | 'downscale_in_infer']
                               1. upscale_in_train(默认)，在训练时放大输出
                                  - 训练：out = input * mask / (1.0 - p)
                                  - 推理：out = input
                               2. downscale_in_infer，推理时减小输出
                                  - 训练：out = input * mask
                                  - 推理：out = input * (1.0 - p)
    - **trans_qkvw** (bool，可选) - 是否对 qkv 的权重进行转置。
            如果为 true，则 qkv 的权重形状应为[3, num_head, dim_head, dim_embed]。
            否则，qkv 的权重形状应为[dim_embed, 3, num_head, dim_head]。默认为 True。
    - **ring_id** (int，可选) - 用于张量模型并行中的分布式前向传播，仅支持 NCCL。默认为-1，表示不使用 mp。
    - **name** (str，可选) - 操作的名称（可选，默认为 None）。更多信息，请参阅 :ref:`api_guide_Name`。

返回
::::::::::::
    - Tensor|tuple：如果 ``cache_kvs`` 为 None，则返回与 ``x`` 形状和数据类型相同的张量，代表 Transformer 的输出。如果 ``cache_kvs`` 不为 None，则返回元组（output, cache_kvs），其中 output 是 Transformer 的输出，cache_kvs 与输入`cache_kvs`原地更新。

代码示例
:::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_multi_transformer
