.. _cn_api_paddle_nn_attention_flex_attention_or_masks:

or_masks
-------------------------------

.. py:function:: paddle.nn.attention.flex_attention.or_masks(*mask_mods)

返回一个 mask 函数，对输入的多个 mask 函数结果进行逻辑或运算。

参数
::::::::::::

    - **mask_mods** (Callable) - mask 函数，签名为 ``mask_mod(b, h, q_idx, kv_idx)``。

返回
::::::::::::

Callable：对所有 mask 结果执行逻辑或运算后的 mask 函数。

代码示例
::::::::::::

COPY-FROM: paddle.nn.attention.flex_attention.or_masks
