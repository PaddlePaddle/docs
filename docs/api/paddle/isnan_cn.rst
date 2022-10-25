.. _cn_api_tensor_isnan:

isnan
-----------------------------

.. py:function:: paddle.isnan(x, name=None)

返回输入 tensor 的每一个值是否为 `+/-NaN` 。

参数
:::::::::
    - **x** (Tensor) - 输入的 `Tensor`，数据类型为：float16、float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，每个元素是一个 bool 值，表示输入 `x` 的每个元素是否为 `+/-NaN` 。

代码示例
:::::::::

COPY-FROM: paddle.isnan
