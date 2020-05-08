.. _cn_api_tensor_mm:

mm
-------------------------------

.. py:function:: paddle.mm(input, mat2, out=None, name=None)

用于两个输入矩阵的相乘。

两个输入的形状可为任意维度，但当任一输入维度大于3时，两个输入的维度必须相等。

如果原始 Tensor input 或 mat2 的秩为 1 且未转置，则矩阵相乘后的前置或附加维度 1 将移除。

参数：
    - **input** (Variable) : 输入变量，类型为 Tensor 或 LoDTensor。
    - **mat2** (Variable) : 输入变量，类型为 Tensor 或 LoDTensor。
    - **out** (Variable, 可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
    - Variable (Tensor / LoDTensor)，矩阵相乘后的结果。

返回类型：
    - Variable（变量）。

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

    * 例 7:

    input: [M], mat2: [N]
    out: [M, N]


**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid

    input = fluid.data(name='input', shape=[2, 3], dtype='float32')
    mat2 = fluid.data(name='mat2', shape=[3, 2], dtype='float32')
    out = paddle.mm(input, mat2) # out shape is [2, 2]

