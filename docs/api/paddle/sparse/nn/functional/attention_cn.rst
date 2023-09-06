.. _cn_api_paddle_sparse_nn_functional_attention:

attention
-------------------------------
.. py:function:: paddle.sparse.nn.functional.attention(query, key, value, sparse_mask, key_padding_mask=None, attn_mask=None, name=None)

.. note::
    该 API 从 `CUDA 11.7` 开始支持。

稀疏 Attention，该 API 内部使用 SparseCsrTensor 来存储 Transformer 模块中的 attention 矩阵，从而达到减少显存占用、提高性能的目的。
参数 `sparse_mask` 描述了稀疏矩阵的非 0 元素索引布局。

.. math::
    result = softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

其中：矩阵 `Q` `K` `V` 表示 attention 模块的三个输入 Tensor，其 shape 均为 `[batch_size, num_heads, seq_len, head_dim]` ，
公式中的 `d` 代表 `head_dim` 。

参数
::::::::::
    - **query** (DenseTensor) - Attention 模块的 `query` 输入，4D Tensor，数据类型为 float32、float64。
    - **key** (DenseTensor) - Attention 模块的 `key` 输入，4D Tensor，数据类型为 float32、float64。
    - **value** (DenseTensor) - Attention 模块的 `value` 输入，4D Tensor，数据类型为 float32、float64。
    - **sparse_mask** (SparseCsrTensor) - Attention 模块的非 0 元素布局，是一个 3D 的 SparseCsrTensor，shape 为 `[batch_size*num_heads, seq_len, seq_len]` 。
      同时每个批次的非 0 元素个数均相等。`crows` 和 `cols` 的数据类型为 int64，`value` 的数据类型为 float32、float64。
    - **key_padding_mask** (DenseTensor, 可选) - Attention 模块中的 key padding mask，是一个 2D 的 DenseTensor，shape 为 `[batch_size, seq_len]` 。
      数据类型为 float32、float64。默认：None，表示无此掩码运算。
    - **attn_mask** (DenseTensor, 可选) - Attention 模块中的 attention mask，是一个 2D 的 DenseTensor，shape 为 `[seq_len, seq_len]` 。
      数据类型为 float32、float64。默认：None，表示无此掩码运算。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
DenseTensor: 维度为 4，shape 为 `[batch_size, num_heads, seq_len, head_dim]` ，dtype 与输入相同。


代码示例
:::::::::

COPY-FROM: paddle.sparse.nn.functional.attention
