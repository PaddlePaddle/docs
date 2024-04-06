.. _cn_api_paddle_log10:

log10
-------------------------------

.. py:function:: paddle.log10(x, name=None)





Log10 激活函数（逐元素计算底为 10 的对数）

.. math::
                  \\Out=log_{10} x\\


参数
::::::::::::

  - **x** (Tensor) – 输入的 Tensor。数据类型为 int32、int64、float16、bfloat16、float32、float64、complex64 或 complex128。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 Tensor，Log10 算子底为 10 对数输出，数据类型与输入一致。


代码示例
::::::::::::

COPY-FROM: paddle.log10
