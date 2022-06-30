.. _cn_api_sparse_attention:
sparse_attention
-------------------------------

.. py:function:: paddle.nn.functional.sparse_attention(query, key, value, sparse_csr_offset, sparse_csr_columns, name=None)


该OP对Transformer模块中的Attention矩阵进行了稀疏化，从而减少内存消耗和计算量。

其稀疏数据排布通过CSR格式表示，CSR格式包含两个参数，``offset`` 和 ``colunms``。计算公式为：

.. math::
   result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

其中，``Q``，``K``，``V`` 表示注意力模块的三个输入参数。这三个参数的维度是一样的。``d`` 代表这三个参数的最后一个维度的大小。

.. warning::
    目前该API只在CUDA11.3及以上版本中使用。

参数
:::::::::
  - query (Tensor) - 输入的Tensor，代表注意力模块中的 ``query``，这是一个4维Tensor，形状为：[batch_size, num_heads, seq_len, head_dim]，数据类型为float32或float64。
  - key (Tensor) - 输入的Tensor，代表注意力模块中的 ``key``，这是一个4维Tensor，形状为：[batch_size, num_heads, seq_len, head_dim]，数据类型为float32或float64。
  - value (Tensor) - 输入的Tensor，代表注意力模块中的 ``value``，这是一个4维Tensor，形状为：[batch_size, num_heads, seq_len, head_dim]，数据类型为float32或float64。
  - sparse_csr_offset (Tensor) - 输入的Tensor，注意力模块中的稀疏特性，稀疏特性使用CSR格式表示，``offset`` 代表矩阵中每一行非零元的数量。这是一个3维Tensor，形状为：[batch_size, num_heads, seq_len + 1]，数据类型为int32。
  - sparse_csr_columns (Tensor) - 输入的Tensor，注意力模块中的稀疏特性，稀疏特性使用CSR格式表示，``colunms`` 代表矩阵中每一行非零元的列索引值。这是一个3维Tensor，形状为：[batch_size, num_heads, sparse_nnz]，数据类型为int32。

返回
:::::::::
  ``Tensor``，代表注意力模块的结果。这是一个4维Tensor，形状为：[batch_size, num_heads, seq_len, head_dim]，数据类型为float32或float64。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.sparse_attention