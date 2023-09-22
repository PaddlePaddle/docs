.. _cn_api_paddle_incubate_nn_functional_fused_multi_head_attention:

fused_multi_head_attention
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_multi_head_attention(x, qkv_weight, linear_weight, pre_layer_norm=False, pre_ln_scale=None, pre_ln_bias=None, ln_scale=None, ln_bias=None, pre_ln_epsilon=1e-05, qkv_bias=None, linear_bias=None, cache_kv=None, attn_mask=None, dropout_rate=0.5, attn_dropout_rate=0.5, ln_epsilon=1e-05, training=True, mode='upscale_in_train', ring_id=-1, add_residual=True, num_heads=-1, transpose_qkv_wb=False, name=None)

**多头注意力机制**

注意力机制可以将查询（Query）与一组键值对（Key-Value）映射到输出。而多头注意力机制是将注意力机制的计算过程计算多次，以便模型提取不同子空间的信息。

细节可参考论文 `Attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 。

fused_multi_head_attention 算子目前只支持在 GPU 下运行，其包含的计算功能如下：

.. code-block:: ipython

    # pseudocode
    if pre_layer_norm:
      out = layer_norm(x)
      out = linear(out) + qkv) + bias
    else:
      out = linear(x) + bias
    out = transpose(out, perm=[2, 0, 3, 1, 4])
    # extract q, k and v from out.
    q = out[0:1,::]
    k = out[1:2,::]
    v = out[2:3,::]
    out = q * k^t
    out = attn_mask + out
    out = softmax(out)
    out = dropout(out)
    out = out * v
    out = transpose(out, perm=[0, 2, 1, 3])
    out = out_linear(out)
    if pre_layer_norm:
        out = x + dropout(linear_bias + out)
    else:
        out = layer_norm(x + dropout(linear_bias + out))


值得注意的是，该 API 中，q, k, v 的 weight 被统一存储在一个权重 Tensor 中，形状为 `[3, num_heads, head_dim, embed_dim]` ,
如果想得到单独的 q, k 或 v 的 weight，可以通过转置和切分得到。


参数
::::::::::::

::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，代表 Query，是一个三维 tensor，形状为 ``[batch_size, sequence_length, embed_dim]``。其中，batch_size 是一次训练所处理的样本个数（句子个数)；sequence_length 代表每一个样本序列（每句话）中的 word 个数；embed_dim 代表 word 经过 embedding 后得到的向量长度。
    - **qkv_weight** (Tensor) - 代表 Attention 中计算 q, k, v 时的权重，是一个四维 tensor，当 ``transpose_qkv_wb`` 为 False 时形状为 ``[3, num_heads, head_dim, embed_dim]``。其中，3 代表 qkv_weight 是包含了 q, k, v 三个权重矩阵，num_heads 代表 multi-head attention 中的 head 数量，head_dim 代表 head 的维度。当 ``transpose_qkv_wb`` 为 True 时形状为 ``[embed_dim， 3 * embed_dim]``。
    - **linear_weight** (Tensor) - 代表 linear 的权重，二维 tensor，形状为 ``[embed_dim, embed_dim]`` 。
    - **pre_layer_norm** (bool，可选) - 代表是采用 pre_layer_norm 的结构（True）还是 post_layer_norm 的结构（False）。若为 True，则为 pre_layer_norm 结构，代表在 multi-head attention 和 ffn 之前各执行一次 ``layer_norm``。若为 False，则为 post_layer_norm 结构，代表在 multi-head attention 和 ffn 之后各执行一次 ``layer_norm``。默认值：``False`` 。
    - **pre_ln_scale** (Tensor，可选) - 代表 normalize_before 为 True 时，multi-head attention 中第一个 ``layer_norm`` 的权重，一维 tensor，形状为 ``[embed_dim]`` 。
    - **pre_ln_bias** (Tensor，可选) - 代表 normalize_before 为 True 时，multi_head attention 中第一个 ``layer_norm`` 的偏置，一维 tensor，形状为  ``[embed_dim]`` 。
    - **ln_scale** (Tensor，可选) - 代表 normalize_before 为 True 时，multi-head attention 中第二个 （False 时的第一个） ``layer_norm`` 的权重，一维 tensor，形状为 ``[embed_dim]`` 。
    - **ln_bias** (Tensor，可选) - 代表 normalize_before 为 True 时，multi-head attention 中第二个 （False 时的第一个） ``layer_norm`` 的偏置，一维 tensor，形状为 ``[embed_dim]`` 。
    - **pre_ln_epsilon** (float，可选) - 代表 normalize_before 为 True 时，multi-head attention 中第一个 ``layer_norm`` 为了数值稳定加在分母上的值。默认值为 1e-05 。
    - **qkv_bias** (Tensor，可选) - 代表 Attention 中计算 q, k, v 时的偏置，是一个三维 tensor，当 ``transpose_qkv_wb`` 为 False 时形状为 ``[3, num_heads, head_dim]`` 。当 ``transpose_qkv_wb`` 为 True 时形状为 ``[3 * embed_dim]`` 。
    - **linear_bias** (Tensor，可选) - 代表 ``linear`` 的偏置，一维 tensor，形状为 ``[embed_dim]`` 。
    - **cache_kv** (Tensor，可选) - 代表自回归生成模型中 cache 结构的部分，五维 tensor，形状为 ``[2, bsz, num_head, seq_len, head_dim]``。默认值为 None。
    - **attn_mask** （Tensor，可选）- 用于限制 multi-head attention 中对当前词产生影响的其他词的范围。形状会被广播为 ``[batch_size, num_heads, sequence_length, sequence_length ]`` 。
    - **dropout_rate** (float，可选) - 代表 multi-head attention 之后的 dropout 算子的 dropout 比例，默认为 0.5。
    - **attn_dropout_rate** (float，可选) - 代表 multi-head attention 中的 dropout 算子的 dropout 比例，默认为 0.5。
    - **ln_epsilon** (float，可选) - 代表 normalize_before 为 True 时，multi-head attention 中第二个 （False 时的第一个） ``layer_norm`` 为了数值稳定加在分母上的值。默认值为 1e-05 。
    - **training** (bool)：标记是否为训练阶段。默认：True。
    - **mode** (str)：丢弃单元的方式，有两种'upscale_in_train'和'downscale_in_infer'，默认：'upscale_in_train'。计算方法如下：

        1. upscale_in_train，在训练时增大输出结果。

            - train: out = input * mask / ( 1.0 - p )
            - inference: out = input

        2. downscale_in_infer，在预测时减小输出结果

            - train: out = input * mask
            - inference: out = input * (1.0 - p)
    - **ring_id** (int，可选) - 分布式 tensor parallel 运行下通讯所使用的 NCCL id。默认值为 -1 。
    - **add_residual** (bool，可选) - 是否在计算最后对结果进行残差计算。默认值为 True。
    - **num_heads** (int，可选) - 在 ``transpose_qkv_wb`` 设置为 True 的时候，必须提供该值，表示 Multi-Head Attention 的 head 的维度。默认值为 -1。
    - **transpose_qkv_wb** (bool，可选) - 是否在底层算子中对 Attention 中计算 q, k, v 时的权重与偏置进行 transpose 操作。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
:::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_multi_head_attention
