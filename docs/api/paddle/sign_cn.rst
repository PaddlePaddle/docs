.. _cn_api_paddle_sign:

sign
-------------------------------

.. py:function:: paddle.sign(x, name=None)

对输入参数 ``x`` 中每个元素进行判断，并且输出判断值：对于实数， 1 表示正数，-1 表示负数，0 表示零。对于复数，返回值是一个单位大小的复数。如果复数元素为零，则返回 0+0j。

参数
::::::::::::
    - **x** (Tensor) – 进行正负值判断的多维 Tensor，数据类型为 uint8, int8， int16， int32， int64， bfloat16， float16， float32， float64， complex64 或 complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，输出正负号，数据的 shape 大小及数据类型和输入 ``x`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.sign
