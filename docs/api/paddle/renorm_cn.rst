.. _cn_api_paddle_renorm:

renorm
------------------------

.. py:function:: paddle.renorm(x, p, axis, max_norm)

此操作用于沿指定轴计算 p 范数（p-norm），假设指定轴维度的形状值为 T，则将张量分为 T 个部分，每个部分都计算 p 范数。如果第 i 部分的 p-norm 大于 max-norm，则第 i 部分中的每个元素应以相同的比例重新归一化，以使第 i 部分的 p-norm 完全等于 max-norm；否则，第 i 部分保持不变。 


参数 
::::::::::::
- **x** (Tensor) - 输入张量
- **p** (float) - 范数运算的幂。
- **axis** (int) - 对张量进行切片的维度。
- **max-norm** (float) - 最大范数限制。

返回 
::::::::::::::

Tensor:renorm Tensor

代码示例 
::::::::::::

COPY-FROM: paddle.renorm