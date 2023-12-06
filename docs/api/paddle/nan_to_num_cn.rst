.. _cn_api_paddle_nan_to_num:

nan_to_num
-------------------------------

.. py:function:: paddle.nan_to_num(x, nan=0.0, posinf=None, neginf=None, name=None)

替换 x 中的 NaN、+inf、-inf 为指定值。

参数
:::::::::
    - **x** (Tensor) - 输入变量，类型为 Tensor， 支持 float32、float64 数据类型。
    - **nan** (float，可选) - NaN 的替换值，默认为 0。
    - **posinf** (float，可选) - +inf 的替换值，默认为 None，表示使用输入 Tensor 的数据类型所能表示的最大值作为 +inf 的替换值。
    - **neginf** (float，可选) - -inf 的替换值，默认为 None，表示使用输入 Tensor 的数据类型所能表示的最小值作为 -inf 的替换值。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
:::::::::
    - Tensor (Tensor)，将输入 Tensor 中的 NaN、+inf、-inf 替换后的结果。


代码示例
:::::::::

COPY-FROM: paddle.nan_to_num
