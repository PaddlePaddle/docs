.. _cn_api_finfo:

finfo
-------------------------------

.. py:function:: paddle.finfo(dtype)



返回一个 ``finfo`` 对象，该对象包含了输入 ``dtype`` 的各种相关的数值信息。其中输入 ``dtype`` 只能是整数类型的 ``paddle.dtype`` 。

其功能类似 `numpy.finfo <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html#numpy-finfo>`_ 。


参数
:::::::::
    - **dtype** (paddle.dtype) - 输入的数据类型，只能为：paddle.float16、 paddle.float32、 paddle.float64、 paddle.bfloat16、 paddle.complex64 和 paddle.complex128 。

返回
:::::::::
一个 ``finfo`` 对象，其中包含 8 个属性，如下所示：

    - **min** (double) - 该数据类型所能表示的最小的数。
    - **max** (double) - 该数据类型所能表示的最大的数。
    - **eps** (double) - 该数据类型所能表示的最小数，使得 1.0 + eps ≠ 1.0 。
    - **resolution** (double) - 这种类型的近似小数分辨率。
    - **smallest_normal** (double) - 这种类型的最小的正 normal 数。
    - **tiny** (double) - 这种类型的最小的正 normal 数，和 smallest_normal 相同。
    - **bits** (int) - 该数据类型所占据的 bit 位数。
    - **dtype** (str) - 该数据类型的字符串名称。


代码示例
:::::::::

COPY-FROM: paddle.finfo
