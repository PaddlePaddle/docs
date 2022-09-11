.. _cn_api_paddle_incubate_sparse_transpose:

transpose
-------------------------------

.. py:function:: paddle.incubate.sparse.transpose(x, dims, name=None)



将输入 :attr:`x` 的维度按照 :attr:`dims` 给定的顺序进行重排，不改变数据值。

输入 :attr:`x` 的维度与输入 :attr:`dims` 的长度必须相同，且输入 :attr:`dims` 必须包含全部维度。


参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为 float32、float64、int32 或 int64。
    - **dims** (list|tuple) - 给定的重排顺序。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和压缩格式与 :attr:`x` 相同。


代码示例
:::::::::

COPY-FROM: paddle.incubate.sparse.transpose
