.. _cn_api_paddle_index_put:

index_put
-------------------------------

.. py:function::  paddle.index_put(x, indices, value, accumulate=False, name=None)

依据索引 ``indices`` ，将指定位置的 ``x`` 重新赋值为 ``value`` 。这里 ``indices`` 是一个 ``tuple of tensor`` 。

参数
:::::::::

    - **x** （Tensor）– 输入 Tensor。 ``x`` 的数据类型可以是 float16, float32，float64，int32，int64，bool。
    - **indices** （Tuple of Tensor）– 包含用来索引的 tensors 的元组。数据类型为 int32，int64，bool。
    - **value**    (Tensor) – 用来给 ``x`` 赋值的 Tensor。
    - **accumulate** （Tensor，可选）– 指定是否将 ``value`` 加到 ``x`` 的参数。 默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，返回一个数据类型同输入的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.index_put

更多关于 outplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位（Outplace）操作的区别`_ 了解详情。

.. _3.1.3 原位（Inplace）操作和非原位（Outplace）操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3
