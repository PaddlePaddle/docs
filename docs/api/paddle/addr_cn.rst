.. _cn_api_paddle_addr:

addr
-------------------------------

.. py:function:: paddle.addr(input, vec1, vec2, beta=1, alpha=1, *, out=None)

执行向量 ``vec1`` 和向量 ``vec2`` 的外积，并将其加到输入矩阵上。

公式为：out = beta * input + alpha * (vec1 outer vec2)

参数
:::::::::
    - **input** (Tensor) - 待加的输入 Tensor。
    - **vec1** (Tensor) - 第一个向量。
    - **vec2** (Tensor) - 第二个向量。
    - **beta** (float，可选) - input 的乘数，默认值为 1。
    - **alpha** (float，可选) - 外积的乘数，默认值为 1。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
Tensor：计算结果 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.addr
