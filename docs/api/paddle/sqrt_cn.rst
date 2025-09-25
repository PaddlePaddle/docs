.. _cn_api_paddle_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.sqrt(x, name=None, *, out=None)




计算输入的算数平方根。

.. math:: out=\sqrt x=x^{1/2}

.. note::
    别名支持: 参数名  ``input``  可替代  ``x`` ；

.. note::
    请确保输入中的数值是非负数。

参数
::::::::::::


    - **x** (Tensor) - 支持任意维度的 Tensor。数据类型为 float16，float32，float64，complex64 或 complex128。
       ``别名: input`` 
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **out** (Tensor，可选) - 关键字参数。输出 Tensor，用于存储计算结果。如果指定，则结果将写入此 Tensor 中。

返回
::::::::::::
返回类型为 Tensor，数据类型同输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.sqrt
