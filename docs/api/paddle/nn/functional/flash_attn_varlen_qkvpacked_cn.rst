.. _cn_api_paddle_nn_functional_flash_attn_varlen_qkvpacked:

flash_attn_varlen_qkvpacked
-------------------------------

.. py:function:: paddle.nn.functional.flash_attn_varlen_qkvpacked(qkv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, dropout=0.0, causal=False, return_softmax=False, return_softmax_lse=False, return_seed_offset=False, varlen_padded=True, training=True, name=None)

支持变长batch的QKVPacked的 flash_attention。

.. math::

    result = softmax(\frac{ Q * K^T }{\sqrt{d}} + mask) * V

参数
::::::::::::

    - **qkv** (Tensor) - 输入 Query+Key+Value Tensor，shape =[total_seqlen, num_group + 2, num_heads_k, head_dim]，数据类型为 float16 或 bfloat16，其中num_group=num_heads_q/num_heads_k，并且qkv[:,:num_group,:,:]是Query Tensor，qkv[:,num_group,:,:]是Key Tensor，qkv[:,num_group+1,:,:]是Value Tensor
    - **cu_seqlens_q** (Tensor) - batch中Query序列的累积序列长度
    - **cu_seqlens_k** (Tensor) - batch中Key/Value序列的累积序列长度
    - **max_seqlen_q** (int) - batch中单个Query序列最大长度
    - **max_seqlen_k** (int) - batch中单个Key/Value序列最大长度
    - **scale** (float) - QK^T在执行Softmax前的缩放因子
    - **dropout** (bool，可选) – dropout 概率值，默认值为 0。
    - **causal** (bool，可选) - 是否使用 causal 模式，默认值：False。
    - **return_softmax** (bool，可选) - 是否返回 softmax 的结果。默认值 False。
    - **return_softmax_lse** (bool，可选) - 是否返回 return_softmax_lse 的结果。默认值为 False。
    - **return_seed_offset** (bool，可选) - 是否返回 return_seed_offset 的结果。默认值为 False。
    - **varlen_padded** (bool，可选) - 输入输出是否采用Padded模式，当设置为True时输入输出应按照max_seqlen序列长度Padding，否则输入输出为Unpad格式，默认值为 True
    - **training** (bool，可选) - 指示是否为训练模式。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
`Tensor`，attention 的结果。


代码示例
::::::::::::
COPY-FROM: paddle.nn.functional.flash_attn_varlen_qkvpacked
