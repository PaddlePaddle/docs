.. _cn_api_paddle_nn_functional_flash_attn_varlen_qkvpacked:

flash_attn_varlen_qkvpacked
-------------------------------

.. py:function:: paddle.nn.functional.flash_attn_varlen_qkvpacked(qkv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, dropout=0.0, causal=False, return_softmax=False, fixed_seed_offset=None, rng_name="", varlen_padded=True, training=True, name=None)

支持变长 batch 的 QKVPacked 的 flash_attention。

.. math::

    result = softmax(\frac{ Q * K^T }{\sqrt{d}} + mask) * V

参数
::::::::::::

    - **qkv** (Tensor) - 输入 Query+Key+Value Tensor，shape =[total_seqlen, num_group + 2, num_heads_k, head_dim]，数据类型为 float16 或 bfloat16，其中 num_group=num_heads_q/num_heads_k，并且 qkv[:,:num_group,:,:]是 Query Tensor，qkv[:,num_group,:,:]是 Key Tensor，qkv[:,num_group+1,:,:]是 Value Tensor
    - **cu_seqlens_q** (Tensor) - batch 中 Query 序列的累积序列长度
    - **cu_seqlens_k** (Tensor) - batch 中 Key/Value 序列的累积序列长度
    - **max_seqlen_q** (int) - batch 中单个 Query 序列最大长度
    - **max_seqlen_k** (int) - batch 中单个 Key/Value 序列最大长度
    - **scale** (float) - QK^T 在执行 Softmax 前的缩放因子
    - **dropout** (bool，可选) – dropout 概率值，默认值为 0。
    - **causal** (bool，可选) - 是否使用 causal 模式，默认值：False。
    - **return_softmax** (bool，可选) - 是否返回 softmax 的结果。默认值 False。
    - **fixed_seed_offset** (Tensor，可选) - Dropout mask 的偏移量，默认值：None。
    - **rng_name** (str，可选) - 随机数生成器的名称。默认值：""。
    - **varlen_padded** (bool，可选) - 输入输出是否采用 Padded 模式，当设置为 True 时输入输出应按照 max_seqlen 序列长度 Padding，否则输入输出为 Unpad 格式，默认值为 True
    - **training** (bool，可选) - 指示是否为训练模式。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
`Tensor`，attention 的结果。


代码示例
::::::::::::
COPY-FROM: paddle.nn.functional.flash_attn_varlen_qkvpacked
