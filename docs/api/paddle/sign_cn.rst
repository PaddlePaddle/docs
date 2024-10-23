.. _cn_api_paddle_sign:

sign
-------------------------------

.. py:function:: paddle.sign(x, name=None)

对输入参数 ``x`` 中每个元素进行正负判断，并且输出正负判断值：1 代表正，-1 代表负，0 代表零。

参数
::::::::::::
    - **x** (Tensor) – 进行正负值判断的多维 Tensor，数据类型为 uint8, int8， int16， int32， int64， bfloat16， float16， float32 或 float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，输出正负号，数据的 shape 大小及数据类型和输入 ``x`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.sign
