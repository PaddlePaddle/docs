.. _cn_api_paddle_compat_equal:

equal
-------------------------------

.. py:function:: paddle.compat.equal(input, other)

判断两个 Tensor 的形状和全部元素是否相同。该兼容接口返回单个 Python ``bool``，不会因为输入 dtype 不同而直接判定不相等；包含 NaN 的 Tensor 不与自身相等。

参数
:::::::::

    - **input** (Tensor) - 第一个输入 Tensor。
    - **other** (Tensor) - 第二个输入 Tensor。

返回
:::::::::

bool，形状和全部元素相同时为 True。

代码示例
:::::::::

COPY-FROM: paddle.compat.equal

