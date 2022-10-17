.. _cn_api_paddle_incubate_sparse_coalesce:

coalesce
-------------------------------

.. py:function:: paddle.incubate.sparse.coalesce(x, name=None)

coalesce 操作包含排序和合并相同 indices 两步，执行 coalesce 后，x 变成按 indices 进行有序排序，并行每个 indices 只出现一次。

参数
:::::::::
    - **x** (Tensor) - 输入 SparseCooTensor

返回
:::::::::
返回 coalesce 后的 SparseCooTensor。

代码示例
:::::::::

COPY-FROM: paddle.incubate.sparse.add
