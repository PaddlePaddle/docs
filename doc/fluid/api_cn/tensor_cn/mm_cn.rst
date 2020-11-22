.. _cn_api_tensor_mm:

mm
-------------------------------

.. py:function:: paddle.mm(input, mat2, out=None, name=None)

:update_api: paddle.matmul



用于两个输入矩阵的相乘。

两个输入的形状可为任意维度，但当任一输入维度大于3时，两个输入的维度必须相等。

如果原始 Tensor input 或 mat2 的维度为 1 且未转置，则矩阵相乘后的前置或附加维度 1 将移除。

mm不遵循broadcast规则，需要遵循broadcast规则的矩阵相乘请使用paddle.matmul。

参数：
    - **input** (Tensor) : 输入变量，类型为 Tensor 或 LoDTensor。
    - **mat2** (Tensor) : 输入变量，类型为 Tensor 或 LoDTensor。
    - **out** (Tensor, 可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
    - Tensor (Tensor / LoDTensor)，矩阵相乘后的结果。

返回类型：
    - Tensor（变量）。

**代码示例**：

.. code-block:: python

    import paddle

    input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
    mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
    out = paddle.mm(input, mat2)
    print(out)
    # Tensor(shape=[3, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
    #        [[11., 14., 17., 20.],
    #         [23., 30., 37., 44.],
    #         [35., 46., 57., 68.]])
