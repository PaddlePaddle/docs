.. _cn_api_paddle_from_numpy:

from_numpy
-------------------------------


.. py:function:: paddle.from_numpy(ndarray)

从 ``ndarray`` 构造一个 Tensor。

返回的 Tensor 与 ``ndarray`` 共享同一片内存。


参数
:::::::::

    - **ndarray** (ndarray) - 待转换的 ``ndarray``。
返回
:::::::::
与 ``ndarray`` 共享同一片内存的 Tensor。


代码示例
:::::::::

COPY-FROM: paddle.from_numpy
