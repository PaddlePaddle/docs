.. _cn_api_paddle_Tensor_sparse_mask:

sparse_mask
-------------------------------

.. py:method:: paddle.Tensor.sparse_mask(self, mask, name=None)

将当前稠密 Tensor 通过稀疏掩码（sparse mask）进行掩码操作，生成新的稀疏 Tensor。输出稀疏 Tensor 的索引与 `mask` 一致，值从当前 Tensor 对应位置提取。

**说明**:
    该方法是 `paddle.sparse.mask_as` 的便捷封装，等价于 `paddle.sparse.mask_as(self, mask)`。

**参数**:
    - **self** (Tensor) - 输入的稠密 Tensor（将被过滤）。
    - **mask** (Tensor) - 用于掩码的稀疏 Tensor（`SparseCooTensor` 或 `SparseCsrTensor`）。
    - **name** (str, 可选) - 操作名称（在实现中被忽略，不生效）。

**返回**:
    SparseTensor - 新生成的稀疏 Tensor，索引与 `mask` 相同，值来自 `self` 在掩码位置的元素。

**代码示例**:

.. code-block:: python

    >>> import paddle
    >>> paddle.set_device('cpu')
    >>>
    >>> # 创建 CSR 格式稀疏掩码
    >>> crows = [0, 2, 3, 5]
    >>> cols = [1, 3, 2, 0, 1]
    >>> values = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> dense_shape = [3, 4]
    >>> csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
    >>>
    >>> # 创建稠密输入张量
    >>> x = paddle.rand(dense_shape)
    >>>
    >>> # 通过稀疏掩码生成稀疏输出
    >>> out = x.sparse_mask(csr)
    >>> print(out)
    Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,
           crows=[0, 2, 3, 5],
           cols=[1, 3, 2, 0, 1],
           values=[0.23659813, 0.08467803, 0.64152628, 0.66596609, 0.90394485])
