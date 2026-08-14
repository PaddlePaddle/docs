.. _cn_api_paddle_addmv:

addmv
-------------------------------

.. py:function:: paddle.addmv(input, mat, vec, beta=1, alpha=1, *, out=None)

执行矩阵 ``mat`` 和向量 ``vec`` 的矩阵-向量乘法，并将其加到输入 Tensor 上。

公式为：out = beta * input + alpha * (mat @ vec)

参数
:::::::::
    - **input** (Tensor) - 待加的输入 Tensor。
    - **mat** (Tensor) - 待乘的矩阵。
    - **vec** (Tensor) - 待乘的向量。
    - **beta** (float，可选) - input 的乘数，默认值为 1。
    - **alpha** (float，可选) - mat @ vec 的乘数，默认值为 1。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
Tensor：计算结果 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.addmv
