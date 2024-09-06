.. _cn_api_paddle_iinfo:

iinfo
-------------------------------

.. py:function:: paddle.iinfo(dtype)



返回一个 ``iinfo`` 对象，该对象包含了输入 ``dtype`` 的各种相关的数值信息。其中输入 ``dtype`` 只能是整数类型的 ``paddle.dtype`` 。

其功能类似 `numpy.iinfo <https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html#numpy-iinfo>`_ 。


参数
:::::::::
    - **dtype** (paddle.dtype|str) - 输入的数据类型，可以是：paddle.uint8、 paddle.int8、 paddle.int16、 paddle.int32、 paddle.int64 或这些类型的字符串形式。

返回
:::::::::
一个 ``iinfo`` 对象，其中包含 4 个属性，如下所示：

    - **min** (int) - 该数据类型所能表示的最小的整数；
    - **max** (int) - 该数据类型所能表示的最大的整数；
    - **bits** (int) - 该数据类型所占据的 bit 位数；
    - **dtype** (str) - 该数据类型的字符串名称。


代码示例
:::::::::

COPY-FROM: paddle.iinfo
