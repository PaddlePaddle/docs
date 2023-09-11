.. _cn_api_paddle_full_like:

full_like
-------------------------------

.. py:function:: paddle.full_like(x, fill_value, dtype=None, name=None)


创建一个和 ``x`` 具有相同的形状并且数据类型为 ``dtype`` 的 Tensor，其中元素值均为 ``fill_value``，当 ``dtype`` 为 None 的时候，Tensor 数据类型和输入 ``x`` 相同。

参数
::::::::::::

    - **x** (Tensor) – 输入 Tensor，输出 Tensor 和 x 具有相同的形状，x 的数据类型可以是 bool、float16、float32、float64、int32、int64。
    - **fill_value** (bool|float|int) - 用于初始化输出 Tensor 的常量数据的值。注意：该参数不可超过输出变量数据类型的表示范围。
    - **dtype** (np.dtype|str，可选) - 输出变量的数据类型。若参数为 None，则输出变量的数据类型和输入变量相同，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回一个根据 ``x`` 、``fill_value`` 、 ``dtype`` 创建的 Tensor。


代码示例
::::::::::::
COPY-FROM: paddle.full_like
