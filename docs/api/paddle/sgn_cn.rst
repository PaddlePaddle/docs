.. _cn_api_tensor_sgn:

sgn
-------------------------------

.. py:function:: paddle.sgn(x, name=None)

对于复数 Tensor，此函数返回一个新的 Tensor，其元素与 input 元素的角度相同且绝对值为 1。

对于实数 Tensor，对输入参数 ``x`` 中每个元素进行正负判断，并且输出正负判断值：1 代表正，-1 代表负，0 代表零。

参数
::::::::::::
    - **x** (Tensor) – 输入 Tensor，数据类型为 float16、float32、float64、complex64 或 complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，输出正负号或复数的单位向量，数据的 shape 大小及数据类型和输入 ``x`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.sgn
