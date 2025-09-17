.. _cn_api_paddle_log2:

log2
-------------------------------

.. py:function:: paddle.log2(x, name=None, *, out)





Log2 激活函数（计算底为 2 的对数）

.. math::
                  \\Out=log_2x\\


.. note::
    别名支持: 参数名  ``input``  可替代  ``x`` ;

参数
:::::::::
  - **x** (Tensor) – 该 OP 的输入为 Tensor。数据类型为 int32，int64，float16，bfloat16，float32， float64， complex64 或 complex128。
     ``别名：input`` 
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **out** (Tensor，可选) - 输出的结果。该参数为仅关键字参数，默认值为 None。
返回
:::::::::
Tensor,Log2 算子底为 2 对数输出，数据类型与输入一致。


代码示例
:::::::::

COPY-FROM: paddle.log2
