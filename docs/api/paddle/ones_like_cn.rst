.. _cn_api_paddle_ones_like:

ones_like
-------------------------------

.. py:function:: paddle.ones_like(x, dtype=None, name=None)


返回一个和输入参数 ``x`` 具有相同形状的数值都为 1 的 Tensor，数据类型为 ``dtype`` 或者和 ``x`` 相同，如果 ``dtype`` 为 None，则输出 Tensor 的数据类型与 ``x`` 相同。

参数
::::::::::
    - **x** (Tensor) – 输入的 Tensor，数据类型可以是 bool，float16，float32，float64，int32，int64。
    - **dtype** (str|np.dtype，可选) - 输出 Tensor 的数据类型，支持 bool，float16, float32，float64，int32，int64。当该参数值为 None 时，输出 Tensor 的数据类型与 ``x`` 相同。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::

Tensor：和 ``x`` 具有相同形状的数值都为 1 的 Tensor，数据类型为 ``dtype`` 或者和 ``x`` 相同。


代码示例
::::::::::

COPY-FROM: paddle.ones_like
