.. _cn_api_paddle_compat_nn_MultiheadAttention:

MultiheadAttention
-------------------------------

.. py:class:: paddle.compat.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)

PyTorch 兼容的多头注意力层。

参数
:::::::::

    - **embed_dim** (int) - 模型的总特征维度。
    - **num_heads** (int) - 注意力头数。
    - **dropout** (float，可选) - 注意力权重的 dropout 概率。默认值：0.0。
    - **bias** (bool，可选) - 是否为输入和输出投影添加偏置。默认值：True。
    - **add_bias_kv** (bool，可选) - 是否向 key 和 value 序列添加偏置。默认值：False。
    - **add_zero_attn** (bool，可选) - 是否向 key 和 value 序列添加全零项。默认值：False。
    - **kdim** (int|None，可选) - key 的特征维度。默认值：None，使用 ``embed_dim``。
    - **vdim** (int|None，可选) - value 的特征维度。默认值：None，使用 ``embed_dim``。
    - **batch_first** (bool，可选) - 输入输出是否使用 ``[batch, seq, feature]`` 布局。默认值：False。
    - **device** (PlaceLike|None，可选) - 参数设备。默认值：None。
    - **dtype** (DTypeLike|None，可选) - 参数数据类型。默认值：None。

代码示例
:::::::::

COPY-FROM: paddle.compat.nn.MultiheadAttention
