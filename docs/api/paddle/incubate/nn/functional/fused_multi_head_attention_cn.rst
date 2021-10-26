.. _cn_api_incubate_nn_cn_fused_multi_head_attention:

fused_multi_head_attention
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_multi_head_attention(x, qkv_weight, linear_weight, pre_layer_norm=False, pre_ln_scale=None, pre_ln_bias=None, ln_scale=None, ln_bias=None, pre_ln_epsilon=1e-05, qkv_bias=None, linear_bias=None, attn_mask=None, dropout_rate=0.5, attn_dropout_rate=0.5, ln_epsilon=1e-05, name=None)

**多头注意力机制**

注意力机制可以将查询（Query）与一组键值对（Key-Value）映射到输出。而多头注意力机制是将注意力机制的计算过程计算多次，以便模型提取不同子空间的信息。

细节可参考论文 `Attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 。

fused_multi_head_attention 包含的计算逻辑如下：
.. code-block:: python

    # pseudocode
    if pre_layer_norm:
      out = layer_norm(x);
      out = linear(out) + qkv)bias
    else:
      out = linear(x) + bias;
    out = transpose(out, perm=[2, 0, 3, 1, 4]);
    # extract q, k and v from out.
    q = out[0:1,::]
    k = out[1:2,::]
    v = out[2:3,::]
    out = q * k^t;
    out = attn_mask + out;
    out = softmax(out);
    out = dropout(out);
    out = out * v;
    out = transpose(out, perm=[0, 2, 1, 3]);
    out = out_linear(out);
    out = layer_norm(x + dropout(linear_bias + out));

值得注意的是，该API中，q, k, v 的 weight 被统一存储在一个权重张量中，形状为；``[3, num_heads, head_dim, embed_dim]``。
如果想得到单独的q, k 或v的 weight，可以通过转置和切分得到。

参数:
::::::::::
    - **x** (Tensor) - 输入的 ``Tensor`` ，代表 Query，是一个三维 tensor，形状为``[batch_size, sequence_length, embed_dim]``。其中，batch_size 是一次训练所处理的样本个数（句子个数)；sequence_length 代表每一个样本序列（每句话）中的 word 个数；embed_dim 代表 word 经过 embedding 后得到的向量长度。
    - **qkv_weight** (Tensor) - 代表 Attention 中计算 q, k, v 时的权重，是一个四维 tensor，形状为``[3, num_heads, head_dim, embed_dim]``。其中，3 代表 qkv_weight 是包含了 q, k, v 三个权重矩阵，num_heads 代表 multi-head attention 中的 head 数量，head_dim 代表 head 的维度。
    - **linear_weight** (Tensor) - 代表 linear 的权重，二维 tensor，形状为``[embed_dim, embed_dim]``。
    - **normalize_before** (bool, 可选) - 代表是采用 pre_layer_norm 的结构（True）还是 post_layer_norm 的结构（False）。若为True，则为 pre_layer_norm 结构，代表在 multi-head attention 和 ffn 之前各执行一次layer_norm。若为False，则为 post_layer_norm 结构，代表在 multi-head attention 和 ffn 之后各执行一次layer_norm。默认值：``False``。
    - **pre_ln_scale** (Tensor, 可选) - 代表 normalize_before 为True 时， multi-head attention 中第一个 layer_norm 的权重，一维tensor，形状为``[embed_dim]``。
    - **pre_ln_bias** (Tensor, 可选) - 代表 normalize_before 为True 时， multi_head attention 中第一个 layer_norm 的偏置，一维tensor，形状为``[embed_dim]``。
    - **ln_scale** (Tensor, 可选) - 代表 normalize_before 为True 时， multi-head attention 中第二个 （False时的第一个） layer_norm 的权重，一维tensor，形状为``[embed_dim]``。
    - **ln_bias** (Tensor, 可选) - 代表 normalize_before 为True 时， multi-head attention 中第二个 （False时的第一个） layer_norm 的偏置，一维tensor，形状为``[embed_dim]``。
    - **pre_ln_epsilon** (float, 可选) - 代表 normalize_before 为True 时，multi-head attention 中第一个 layer_norm 为了数值稳定加在分母上的值。默认值为 1e-05 。
    - **qkv_bias** (Tensor, 可选) - 代表 Attention 中计算 q, k, v 时的偏置，是一个三维 tensor，形状为``[3, num_heads, head_dim]``。
    - **linear_bias** (Tensor, 可选) - 代表 linear 的偏置，一维tensor，形状为``[embed_dim]``。
    - **attn_mask** （Tensor, 可选）- 用于限制 multi-head attention中对当前词产生影响的其他词的范围。形状会被广播为``[batch_size, num_heads, sequence_length, sequence_length ]``。
    - **dropout_rate** (float, 可选) - 代表 multi-head attention 之后的 dropout 算子的 dropout 比例，默认为0.5。
    - **attn_dropout_rate** (float, 可选) - 代表 multi-head attention 中的 dropout 算子的 dropout 比例，默认为0.5。
    - **ln_epsilon** (float, 可选) - 代表 normalize_before 为True 时，multi-head attention 中第二个 （False时的第一个） layer_norm 为了数值稳定加在分母上的值。默认值为 1e-05 。
    - **name** (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
:::::::::

.. code-block:: python

    # required: gpu            
    import paddle
    import paddle.incubate.nn.functional as F

    # input: [batch_size, seq_len, embed_dim]
    x = paddle.rand(shape=(2, 4, 128), dtype="float32")
    # qkv_weight: [3, num_head, head_dim, embed_dim]
    qkv_weight = paddle.rand(shape=(3, 4, 32, 128), dtype="float32")
    # qkv_bias: [3, num_head, head_dim]
    qkv_bias = paddle.rand(shape=(3, 4, 32), dtype="float32")
    # linear_weight: [embed_dim, embed_dim]
    linear_weight = paddle.rand(shape=(128, 128), dtype="float32")
    # linear_bias: [embed_dim]
    linear_bias = paddle.rand(shape=[128], dtype="float32")
    # self attention mask: [batch_size, num_heads, seq_len, seq_len]
    attn_mask = paddle.rand(shape=(2, 4, 4, 4), dtype="float32")
    # output: [batch_size, seq_len, embed_dim]
    output = F.fused_multi_head_attention(
        x, qkv_weight, linear_weight, False,
        None, None, None, None, 1e-5, qkv_bias,
        linear_bias, attn_mask)
    # [2, 4, 128]
    print(output.shape)
