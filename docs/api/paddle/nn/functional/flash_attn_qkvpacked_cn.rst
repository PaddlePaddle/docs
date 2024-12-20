.. _cn_api_paddle_nn_functional_flash_attn_qkvpacked:

flash_attn_qkvpacked
-------------------------------

.. py:function:: paddle.nn.functional.flash_attn_qkvpacked(qkv, dropout=0.0, causal=False, return_softmax=False, fixed_seed_offset=None, rng_name="", training=True, name=None)

QKVPacked 的 flash_attention。

.. math::

    result = softmax(\frac{ Q * K^T }{\sqrt{d}} + mask) * V

参数
::::::::::::

    - **qkv** (Tensor) - 输入 Query+Key+Value Tensor，shape =[batch_size, seq_len, num_group + 2, num_heads_k, head_dim]，数据类型为 float16 或 bfloat16，其中 num_group=num_heads_q/num_heads_k，并且 qkv[:,:,:num_group,:,:]是 Query Tensor，qkv[:,:,num_group,:,:]是 Key Tensor，qkv[:,:,num_group+1,:,:]是 Value Tensor
    - **dropout** (bool，可选) – dropout 概率值，默认值为 0。
    - **causal** (bool，可选) - 是否使用 causal 模式，默认值：False。
    - **return_softmax** (bool，可选) - 是否返回 softmax 的结果。默认值 False。
    - **fixed_seed_offset** (Tensor，可选) - Dropout mask 的偏移量，默认值：None。
    - **rng_name** (str，可选) - 随机数生成器的名称。默认值：""。
    - **training** (bool，可选) - 指示是否为训练模式。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
`Tensor`，attention 的结果。


代码示例
::::::::::::
COPY-FROM: paddle.nn.functional.flash_attn_qkvpacked
