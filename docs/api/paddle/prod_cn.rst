.. _cn_api_paddle_prod:

prod
-------------------------------

.. py:function:: paddle.prod(x, axis=None, keepdim=False, dtype=None, name=None)



对指定维度上的 Tensor 元素进行求乘积运算，并输出相应的计算结果。

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，数据类型为：bfloat16、float16、float32、float64、int32、int64。
    - **axis** (int|list|tuple，可选) - 求乘积运算的维度。如果是 None，则计算所有元素的乘积并返回包含单个元素的 Tensor，否则该参数必须在 :math:`[-x.ndim, x.ndim)` 范围内。如果 :math:`axis[i] < 0`，则维度将变为 :math:`x.ndim + axis[i]`，默认为 None。
    - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留输入的维度。除非 keepdim 为 True，否则输出 Tensor 的维度将比输入 Tensor 小一维，默认值为 False。
    - **dtype** (str，可选) - 输出 Tensor 的数据类型，支持 int32、int64、float32、float64。如果指定了该参数，那么在执行操作之前，输入 Tensor 将被转换为 dtype 类型。这对于防止数据类型溢出非常有用。若参数为空，则输出变量的数据类型和输入变量相同，默认为：None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输入 Tensor 在指定 axis 上的累乘的结果。


代码示例
::::::::::::

COPY-FROM: paddle.prod
