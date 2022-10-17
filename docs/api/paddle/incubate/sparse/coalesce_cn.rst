.. _cn_api_paddle_incubate_sparse_coalesce:

coalesce
-------------------------------

.. py:function:: paddle.incubate.sparse.coalesce(x, name=None)

coalesce 操作包含排序和合并相同indices两步，执行coalesce后，x 变成按indices进行有序排序，并行每个indices只出现一次。

参数
:::::::::
    - **x** (Tensor) - 输入SparseCooTensor

返回
:::::::::
返回coalesce后的SparseCooTensor。

代码示例
:::::::::

COPY-FROM: paddle.incubate.sparse.add
