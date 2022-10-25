.. _cn_api_paddle_tolist:

tolist
-------------------------------

.. py:function:: paddle.tolist(x)

将 paddle Tensor 转化为 python list，注意：只适用于动态图。

.. code-block:: text



参数
:::::::::

        - **x** (Tensor) - 输入的 `Tensor`，数据类型为：float32、float64、bool、int8、int32、int64。

返回
:::::::::
Tensor 对应结构的 list。



代码示例
::::::::::::

COPY-FROM: paddle.tolist
