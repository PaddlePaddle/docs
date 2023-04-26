.. _cn_api_tensor_matmul:

matmul
-------------------------------

.. py:function:: paddle.matmul(x, y, transpose_x=False, transpose_y=False, name=None)

.. note::
    计算两个 Tensor 的乘积，遵循完整的广播规则，关于广播规则，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

并且其行为与 ``numpy.matmul`` 一致。目前，输入 Tensor 的维数可以是任意数量，``matmul``  可以用于
实现 ``dot`` ， ``matmul`` 和 ``batchmatmul``。实际行为取决于输入 ``x`` 、输入 ``y`` 、 ``transpose_x`` ，
``transpose_y``。具体如下：

- 如果 ``transpose`` 为真，则对应 Tensor 的后两维会转置。如果 Tensor 的一维，则转置无效。假定 ``x`` 是一个 shape=[D] 的一维 Tensor，则 ``x`` 视为 [1, D]。然而，``y`` 是一个 shape=[D]的一维 Tensor，则视为[D, 1]。

乘法行为取决于 ``x`` 和 ``y`` 的尺寸。具体如下：

- 如果两个 Tensor 均为一维，则获得点积结果。

- 如果两个 Tensor 都是二维的，则获得矩阵与矩阵的乘积。

- 如果 ``x`` 是 1 维的，而 ``y`` 是 2 维的，则将 1 放在 ``x`` 维度之前，以进行矩阵乘法。矩阵相乘后，将删除前置尺寸。

- 如果 ``x`` 是 2 维的，而 ``y`` 是 1 维的，获得矩阵与向量的乘积。

- 如果两个输入至少为一维，且至少一个输入为 N 维（其中 N> 2），则将获得批矩阵乘法。如果第一个自变量是一维的，则将 1 放在其维度的前面，以便进行批量矩阵的乘法运算，然后将其删除。如果第二个参数为一维，则将 1 附加到其维度后面，以实现成批矩阵倍数的目的，然后将其删除。根据广播规则广播非矩阵维度（不包括最后两个维度）。例如，如果输入 ``x`` 是（j，1，n，m）Tensor，另一个 ``y`` 是（k，m，p）Tensor，则 out 将是（j，k，n，p）Tensor。

参数
:::::::::
    - **x** (Tensor) - 输入变量，类型为 Tensor，数据类型为 bfloat16， float16， float32， float64。
    - **y** (Tensor) - 输入变量，类型为 Tensor，数据类型为 bfloat16， float16， float32， float64。
    - **transpose_x** (bool，可选) - 相乘前是否转置 x，默认值为 False。
    - **transpose_y** (bool，可选) - 相乘前是否转置 y，默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Tensor，矩阵相乘后的结果，数据类型和输入数据类型一致。

代码示例
::::::::::

COPY-FROM: paddle.matmul
