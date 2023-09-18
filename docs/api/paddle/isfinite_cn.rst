.. _cn_api_paddle_isfinite:

isfinite
-----------------------------

.. py:function:: paddle.isfinite(x, name=None)

返回输入 Tensor 的每一个值是否为有限值（既非 `+/-INF` 也非 `+/-NaN` ）。

参数
:::::::::
    - **x** (Tensor)：输入的 `Tensor`，数据类型为：float16、float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，每个元素是一个 bool 值，表示输入 `x` 的每个元素是否为有限值（既非 `+/-INF` 也非 `+/-NaN` ）。

代码示例
:::::::::

COPY-FROM: paddle.isfinite
