.. _cn_api_paddle_sparse_coalesce:

coalesce
-------------------------------

.. py:function:: paddle.sparse.coalesce(x, name=None)

coalesce 操作包含排序和合并相同 indices 两步，执行 coalesce 后，x 变成按 indices 进行有序排序，并行每个 indices 只出现一次。

参数
:::::::::
    - **x** (Tensor) - 输入 SparseCooTensor
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
返回 coalesce 后的 SparseCooTensor。

代码示例
:::::::::

COPY-FROM: paddle.sparse.coalesce
