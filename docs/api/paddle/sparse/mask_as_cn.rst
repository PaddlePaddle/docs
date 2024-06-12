.. _cn_api_paddle_sparse_mask_as:

mask_as
-------------------------------

.. py:function:: paddle.sparse.mask_as(x, mask, name=None)

使用稀疏张量 `mask` 的索引过滤输入的稠密张量 `x`，并生成相应格式的稀疏张量。输入的 `x` 和 `mask` 必须具有相同的形状，且返回的稀疏张量具有与 `mask` 相同的索引，即使对应的索引中存在 `零` 值。

参数
:::::::::
    - **x** (DenseTensor) - 输入的 DenseTensor。数据类型为 float32，float64，int32，int64，complex64，complex128，int8，int16，float16。
    - **mask** (SparseTensor) - 输入的稀疏张量，可以为 SparseCooTensor、SparseCsrTensor。当其为 SparseCsrTensor 时，应该是 2D 或 3D 的形式。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
SparseTensor: 其稀疏格式、dtype、shape 均与 `mask` 相同。


代码示例
:::::::::

COPY-FROM: paddle.sparse.mask_as
