.. _cn_api_tensor_increment:

increment
-------------------------------

.. py:function:: paddle.increment(x, value=1.0, name=None)




在控制流程中用来让参数``x`` 的数值增加 ``value`` 。

参数
:::::::::

  - **x** (Tensor) – 输入 Tensor，必须始终只有一个元素。支持的数据类型：float32、float64、int32、int64。
  - **value** (float，可选) – ``x`` 的数值增量。默认值为 1.0。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，形状和数据类型同输入 ``x`` 。


代码示例
::::::::::::

COPY-FROM: paddle.increment
