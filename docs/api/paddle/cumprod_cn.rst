.. _cn_api_paddle_cumprod:

cumprod
-------------------------------

.. py:function:: paddle.cumprod(x, dim=None, dtype=None, name=None)



沿给定维度 ``dim`` 计算输入 tensor ``x`` 的累乘。

.. note::
    结果的第一个元素和输入的第一个元素相同。

参数
:::::::::
    - **x** (Tensor) - 累乘的输入，需要进行累乘操作的 tensor。
    - **dim** (int，可选) - 指明需要累乘的维度，取值范围需在[-x.rank,x.rank)之间，其中 x.rank 表示输入 tensor x 的维度，-1 代表最后一维。
    - **dtype** (str，可选) - 输出 tensor 的数据类型，支持 int32、int64、bfloat16、float16、float32、float64、complex64、complex128。如果指定了，那么在执行操作之前，输入的 tensor 将被转换为 dtype 类型。这对于防止数据类型溢出非常有用。默认为：None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，累乘操作的结果。

代码示例
::::::::::

COPY-FROM: paddle.cumprod
