.. _cn_api_paddle_broadcast_tensors:

broadcast_tensors
-------------------------------

.. py:function:: paddle.broadcast_tensors(inputs, name=None)

根据Broadcast规范对一组输入 ``inputs`` 进行Broadcast操作
输入应符合Broadcast规范

.. note::
    如想了解更多Broadcasting内容，请参见 :ref:`cn_user_guide_broadcasting` 。

参数
:::::::::
    - inputs (list(Tensor)|tuple(Tensor)) - 一组输入Tensor，数据类型为：bool、float32、float64、int32或int64。所有的输入Tensor均需要满足rank <= 5。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``list(Tensor)``，一组Broadcast后的 ``Tensor``，其顺序与 ``input`` 一一对应。

代码示例
:::::::::

COPY-FROM: paddle.broadcast_tensors