.. _cn_api_paddle_compat_allclose:

allclose
-------------------------------

.. py:function:: paddle.compat.allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False, name=None)

逐元素比较两个 Tensor 是否在给定容差内相等，并返回单个 Python ``bool``。

参数
:::::::::

    - **input** (Tensor) - 第一个输入 Tensor。
    - **other** (Tensor) - 第二个输入 Tensor。
    - **rtol** (float，可选) - 相对容差。默认值：1e-5。
    - **atol** (float，可选) - 绝对容差。默认值：1e-8。
    - **equal_nan** (bool，可选) - 是否将同一位置的 NaN 视为相等。默认值：False。
    - **name** (str|None，可选) - API 名称。默认值：None。

返回
:::::::::

bool，所有元素均满足容差条件时为 True。

代码示例
:::::::::

COPY-FROM: paddle.compat.allclose

