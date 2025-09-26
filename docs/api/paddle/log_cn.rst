.. _cn_api_paddle_log:

log
-------------------------------

.. py:function:: paddle.log(x, name=None, *, out=None)




Log 激活函数（计算自然对数）

.. math::
                  \\Out=ln(x)\\


参数
::::::::::::

  - **x** (Tensor) – 输入为 Tensor。数据类型只能为 int32，int64，float16，bfloat16，float32， float64， complex64 或 complex128。别名  ``input`` 。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::

    - **out** (Tensor，可选) - 输出 Tensor，若不为  ``None`` ，计算结果将保存在该 Tensor 中，默认值为  ``None`` 。

返回
::::::::::::
Tensor, Log 算子自然对数输出，数据类型与输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.log
