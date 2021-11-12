.. _cn_api_sparse_attention:
sparse_attention
-------------------------------

.. py:function:: paddle.nn.functional.sparse_attention(query, key, value, sparse_csr_offset, sparse_csr_columns, name=None)


该OP对Transformer模块中的Attention矩阵进行了稀疏化，从而减少内存消耗和计算量。

其稀疏数据排布通过CSR格式表示，CSR格式包含两个参数， ``offset`` 和 ``colunms`` 。计算公式为：

.. math::
   result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

其中，``Q``，``K``，``V`` 表示注意力模块的三个输入参数。这三个参数的维度是一样的。 ``d`` 代表这三个参数的最后一个维度的大小。

.. warning::
    目前该API只在CUDA11.3及以上版本中使用。

参数：
:::::::::
  - query (Tensor) - 输入的Tensor，代表注意力模块中的 ``query`` ，这是一个4维Tensor，形状为 :[batch_size, num_heads, seq_len, head_dim]，数据类型为float32或float64。
  - key (Tensor) - 输入的Tensor，代表注意力模块中的 ``key`` ，这是一个4维Tensor，形状为 :[batch_size, num_heads, seq_len, head_dim]，数据类型为float32或float64。
  - value (Tensor) - 输入的Tensor，代表注意力模块中的 ``value`` ，这是一个4维Tensor，形状为 :[batch_size, num_heads, seq_len, head_dim]，数据类型为float32或float64。
  - sparse_csr_offset (Tensor) - 输入的Tensor，注意力模块中的稀疏特性，稀疏特性使用CSR格式表示， ``offset`` 代表矩阵中每一行非零元的数量。这是一个3维Tensor，形状为 :[batch_size, num_heads, seq_len + 1]，数据类型为int32。
  - sparse_csr_columns (Tensor) - 输入的Tensor，注意力模块中的稀疏特性，稀疏特性使用CSR格式表示， ``colunms`` 代表矩阵中每一行非零元的列索引值。这是一个3维Tensor，形状为 :[batch_size, num_heads, sparse_nnz]，数据类型为int32。

返回：
:::::::::
  ``Tensor`` ，代表注意力模块的结果。这是一个4维Tensor，形状为 :[batch_size, num_heads, seq_len, head_dim]，数据类型为float32或float64。

代码示例
::::::::::

..  code-block:: python

    import paddle
    import numpy as np
    
    query_data = np.array([[[[0, 1,], [2, 3],
            [ 0, 1], [2, 3]]]]).astype("float32")
    key_data = np.array([[[[0, 1,], [2, 3],
                    [ 0, 1], [2, 3]]]]).astype("float32")
    value_data = np.array([[[[0, 1,], [2, 3],
                    [ 0, 1], [2, 3]]]]).astype("float32")
    sparse_csr_offset_data = np.array([[[0, 2,
                    4, 6, 8]]]).astype("int32")
    sparse_csr_columns_data = np.array([[[0, 1,
                    0, 1, 2, 3, 2, 3]]]).astype("int32")
    print(query_data.shape)
    # (1, 1, 4, 2)
    print(sparse_csr_offset_data.shape)
    # (1, 1, 5)
    print(sparse_csr_columns_data.shape)
    # (1, 1, 8)
    paddle.disable_static()
    query = paddle.to_tensor(query_data, stop_gradient=False, 
                    place=paddle.CUDAPlace(0))
    key = paddle.to_tensor(key_data, stop_gradient=False, 
                    place=paddle.CUDAPlace(0))
    value = paddle.to_tensor(value_data, stop_gradient=False, 
                    place=paddle.CUDAPlace(0))
    offset = paddle.to_tensor(sparse_csr_offset_data, stop_gradient=False, 
                    place=paddle.CUDAPlace(0))
    columns = paddle.to_tensor(sparse_csr_columns_data, stop_gradient=False, 
                    place=paddle.CUDAPlace(0))
    output = paddle.nn.functional.sparse_attention(query, key, 
                    value, offset, columns)
    print(output)
    
    # [[[[1.60885942, 2.60885954],
    #       [1.99830270, 2.99830270],
    #       [1.60885942, 2.60885954],
    #       [1.99830270, 2.99830270]]]]