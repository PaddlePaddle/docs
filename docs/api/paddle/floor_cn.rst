.. _cn_api_paddle_floor:

floor
-------------------------------

.. py:function:: paddle.floor(x, name=None, *, out=None)




向下取整函数。

.. math::
    out = \left \lfloor x \right \rfloor

参数
::::::::::::

    - **x** - 输入为多维 Tensor。数据类型支持 float32, float64, float16, bfloat16, uint8, int8, int16, int32, int64。别名  ``input`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::

    - **out** (Tensor，可选) - 输出 Tensor，若不为  ``None`` ，计算结果将保存在该 Tensor 中，默认值为  ``None`` 。

返回
::::::::::::
输出为 Tensor，与  ``x``  维度相同、数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.floor
