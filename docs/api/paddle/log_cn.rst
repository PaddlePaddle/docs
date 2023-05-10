.. _cn_api_fluid_layers_log:

log
-------------------------------

.. py:function:: paddle.log(x, name=None)




Log 激活函数（计算自然对数）

.. math::
                  \\Out=ln(x)\\


参数
::::::::::::

  - **x** (Tensor) – 输入为 Tensor。数据类型只能为 float16，float32 或 float64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor, Log 算子自然对数输出，数据类型与输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.log
