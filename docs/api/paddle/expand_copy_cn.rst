.. _cn_api_paddle_expand_copy:

expand_copy
-------------------------------

.. py:function:: paddle.expand_copy(x, shape, name=None)

根据 ``shape`` 指定的形状扩展 ``x``，扩展后返回一个新的 Tensor，不会修改原 Tensor。

这是 :ref:`cn_api_paddle_expand` 的非原位版本，始终返回新的 Tensor 而非视图。

.. note::
    该接口有两个签名：
    1. ``paddle.expand_copy(x, shape, name=None)`` (Paddle 风格)：根据广播语义返回扩展后的新 Tensor。
    2. ``paddle.expand_copy(input, *size)`` (PyTorch 风格)：通过可变长 size 参数返回扩展后的新 Tensor。

参数
:::::::::::
    - **x** (Tensor) - 输入的 Tensor。别名 ``input``。
    - **shape** (tuple|list|Tensor) - 目标扩展形状。维度数必须大于或等于 ``x`` 的维度数。若 shape 为 list 或 tuple，其中的元素值应全为整数或 0-D 或 1-D Tensor（数据类型为 int32）。别名 ``size``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::::
Tensor，数据类型与 ``x`` 相同。

代码示例
:::::::::::

COPY-FROM: paddle.expand_copy
