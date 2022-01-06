.. _cn_api_tensor_mm:

mm
-------------------------------

.. py:function:: paddle.mm(input, mat2, out=None, name=None)




用于两个输入矩阵的相乘。

两个输入的形状可为任意维度，但当任一输入维度大于3时，两个输入的维度必须相等。

如果原始 Tensor input 或 mat2 的秩为 1 且未转置，则矩阵相乘后的前置或附加维度 1 将移除。

参数：
    - **input** (Tensor) : 输入变量，类型为 Tensor 或 LoDTensor。
    - **mat2** (Tensor) : 输入变量，类型为 Tensor 或 LoDTensor。
    - **out** (Tensor, 可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
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


**代码示例**：

.. code-block:: python

    import paddle

    input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
    mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
    out = paddle.mm(input, mat2)
    # Tensor(shape=[3, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
    #        [[11., 14., 17., 20.],
    #         [23., 30., 37., 44.],
    #         [35., 46., 57., 68.]])
