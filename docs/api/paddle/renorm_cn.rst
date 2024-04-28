.. _cn_api_paddle_renorm:

renorm
-------------------------------

.. py:function:: paddle.renorm(x, p, axis, max_norm)




该函数用于沿指定轴计算 p- 范数，并将每个部分的 p- 范数限制在最大范数值内。


参数
::::::::::::


    - **x** (Tensor) - 输入的 Tensor。支持任意维度的 Tensor，数据类型为 float32，float64 或 float16。
    - **p** (float) - 范数操作的指数。
    - **axis** (int) - 用于分割 Tensor 的维度。
    - **max_norm** (float) - 范数的最大限制值。


返回
::::::::::::
返回重新规范化后的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.renorm
