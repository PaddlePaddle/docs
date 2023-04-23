.. _cn_api_paddle_sparse_cast:

cast
-------------------------------

.. py:function:: paddle.sparse.cast(x, index_dtype=None, value_dtype=None, name=None)

输入 :attr:`x` 为 `SparseCooTensor` 或 `SparseCsrTensor` 。将稀疏 Tensor 的 index 转换为 `index_dtype` 类型
（ `SparseCsrTensor` 的 index 指： `crows` 与 `col` ），value 转换为 `value_dtype` 类型，

参数
:::::::::
    - **x** (SparseTensor) - 输入的稀疏 Tensor，可以为 Coo 或 Csr 格式，数据类型为 float32、float64。
    - **index_dtype** (np.dtype|str, optional) - SparseCooTensor 的 index 类型，SparseCsrTensor 的 crows/cols 类型。可以是 uint8，int8，int16，int32，int64。
    - **value_dtype** (np.dtype|str, optional) - SparseCooTensor 或 SparseCsrTensor 的 value 类型。可以是 uint8，int8，int16，int32，int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor，稀疏格式与 :attr:`x` 相同，数据类型为被转换后的类型。


代码示例
:::::::::

COPY-FROM: paddle.sparse.cast
