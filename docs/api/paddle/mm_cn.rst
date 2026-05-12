.. _cn_api_paddle_mm:

mm
-------------------------------

.. py:function:: paddle.mm(input, mat2, name=None, *, out=None)




用于两个输入矩阵的相乘。

两个输入的形状可为任意维度，但当任一输入维度大于 3 时，两个输入的维度必须相等。

如果原始 Tensor input 或 mat2 的秩为 1 且未转置，则矩阵相乘后的前置或附加维度 1 将移除。

参数
::::::::::::

    - **input** (Tensor) - 输入变量，类型为 Tensor。支持的数据类型：bfloat16、float16、float32、float64、int8、int32、int64、complex64、complex128。
    - **mat2** (Tensor) - 输入变量，类型为 Tensor。支持的数据类型：bfloat16、float16、float32、float64、int8、int32、int64、complex64、complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::::

    - Tensor，矩阵相乘后的结果。


::

    * 例 1:

    input: [B, ..., M, K], mat2: [B, ..., K, N]
    out: [B, ..., M, N]

    * 例 2:

    input: [B, M, K], mat2: [B, K, N]
    out: [B, M, N]

    * 例 3:

    input: [B, M, K], mat2: [K, N]
    out: [B, M, N]

    * 例 4:

    input: [M, K], mat2: [K, N]
    out: [M, N]

    * 例 5:

    input: [B, M, K], mat2: [K]
    out: [B, M]

    * 例 6:

    input: [K], mat2: [K]
    out: [1]


代码示例
::::::::::::

COPY-FROM: paddle.mm
