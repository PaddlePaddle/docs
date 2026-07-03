.. _cn_api_paddle_fix:

fix
-------------------------------

.. py:function:: paddle.fix(input, name=None, *, out=None)

返回输入 Tensor 的截断整数值。与 ``paddle.trunc`` 功能相同。

参数
:::::::::
    - **input** (Tensor) - 输入 Tensor，数据类型为 int32、int64、float32、float64。别名 ``x``。
    - **name** (str，可选) - 操作名称，默认值为 None。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
Tensor：截断后的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.fix
