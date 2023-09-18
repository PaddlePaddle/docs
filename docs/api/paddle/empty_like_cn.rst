.. _cn_api_paddle_empty_like:

empty_like
-------------------------------

.. py:function:: paddle.empty_like(x, dtype=None, name=None)


根据参数 ``x`` 的 shape 和数据类型 ``dtype`` 创建未初始化的 Tensor。如果 ``dtype`` 为 None，则 Tensor 的数据类型与 ``x`` 相同。

参数
::::::::::::

    - **x** (Tensor) – 输入 Tensor，输出 Tensor 和 x 具有相同的形状，x 的数据类型可以是 bool、float16、float32、float64、int32、int64。
    - **dtype** （np.dtype|str，可选）- 输出变量的数据类型，可以是 bool、float16、float32、float64、int32、int64。若参数为 None，则输出变量的数据类型和输入变量相同，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回一个根据 ``x`` 和 ``dtype`` 创建并且尚未初始化的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.empty_like
