.. _cn_api_paddle_floor:

floor
-------------------------------

.. py:function:: paddle.floor(x, name=None, *, out=None)




向下取整函数。

.. math::
    out = \left \lfloor x \right \rfloor

参数
::::::::::::

    - **x** - 输入为多维 Tensor。数据类型必须为 float32 或 float64。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::::
输出为 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.floor
