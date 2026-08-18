.. _cn_api_paddle_compat_unique:

unique
-------------------------------

.. py:function:: paddle.compat.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)

返回输入 Tensor 中的唯一元素，并可选返回逆索引和各元素出现次数。

参数
:::::::::

    - **input** (Tensor) - 输入 Tensor。
    - **sorted** (bool，可选) - 是否对唯一元素排序。默认值：True。
    - **return_inverse** (bool，可选) - 是否返回将唯一元素映射回输入的索引。默认值：False。
    - **return_counts** (bool，可选) - 是否返回每个唯一元素的出现次数。默认值：False。
    - **dim** (int|None，可选) - 计算唯一元素的维度。默认值：None，表示将输入展平后计算。

返回
:::::::::

Tensor；当 ``return_inverse`` 或 ``return_counts`` 为 True 时，返回包含所请求结果的 tuple。

代码示例
:::::::::

COPY-FROM: paddle.compat.unique
