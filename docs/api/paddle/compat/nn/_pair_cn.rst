.. _cn_api_paddle_compat_nn__pair:

_pair
-------------------------------

.. py:function:: paddle.compat.nn._pair(x)

用于将输入转换为包含 2 个元素的元组。当输入为单个整数时，返回包含两个相同整数的元组；当输入为列表或元组时，转换为元组并返回。

参数
:::::::::::

    - **x** (int|list|tuple) - 输入值，可以是整数、包含 2 个元素的列表或元组。

返回
:::::::::::

    tuple：包含 2 个元素的元组。

代码示例
::::::::::::

COPY-FROM: paddle.compat.nn._pair
