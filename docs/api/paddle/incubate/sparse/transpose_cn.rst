.. _cn_api_paddle_incubate_sparse_transpose:

transpose
-------------------------------

.. py:function:: paddle.incubate.sparse.transpose(x, perm, name=None)


根据 :attr:`perm` 对输入的多维 Tensor 进行数据重排。
返回多维 Tensor 的第 i 维对应输入 Tensor 的 perm[i]维。

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为 float32、float64、int32 或 int64。
    - **perm** (list|tuple) - :attr:`perm` 长度必须和 :attr:`x` 的维度相同，并依照 :attr:`perm` 中数据进行重排。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和压缩格式与 :attr:`x` 相同。


代码示例
:::::::::

COPY-FROM: paddle.incubate.sparse.transpose
