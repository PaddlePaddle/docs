.. _cn_api_paddle_incubate_sparse_is_same_shape:

is_same_shape
-------------------------------



.. py:function:: paddle.incubate.sparse.is_same_shape(x, y)

该 API 返回两个Tensor形状比较的结果，判断输入 ``x`` 与输入 ``y`` 的形状是否相同，支持 DenseTensor、SparseCsrTensor 与 SparseCooTensor 之间任意两种的形状比较。

参数
:::::::::
    - x (Tensor) - 输入的Tensor，类型为：DenseTensor、SparseCsrTensor 与 SparseCooTensor 之一
    - y (Tensor) - 输入的Tensor，类型为：DenseTensor、SparseCsrTensor 与 SparseCooTensor 之一

返回
:::::::::
两个Tensor形状比较的结果，相同为 True ，不同为 False 。


代码示例
:::::::::

COPY-FROM: paddle.incubate.sparse.is_same_shape