.. _cn_api_paddle_atleast_2d:

atleast_2d
-------------------------------

.. py:function:: paddle.atleast_2d(*inputs, name=None)

将输入转换为张量并返回至少为 ‵‵2`` 维的视图。``2`` 维或更高维的输入会被保留。

参数
::::::::::::

    - **inputs** (Tensor|list(Tensor)) - 一个或多个 Tensor，数据类型为：``float16``, ``float32``, ``float64``, ``int16``, ``int32``, ``int64``, ``int8``, ``uint8``, ``complex64``, ``complex128``, ``bfloat16`` 或 ``bool``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
一个 Tensor，当只有一个输入的时候。
多个 Tensor，当有多个输入的时候。

代码示例
::::::::::::

COPY-FROM: paddle.atleast_2d
