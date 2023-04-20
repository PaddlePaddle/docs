.. _cn_api_paddle_broadcast_tensors:

broadcast_tensors
-------------------------------

.. py:function:: paddle.broadcast_tensors(inputs, name=None)

根据 Broadcast 规范对一组输入 ``inputs`` 进行 Broadcast 操作，输入应符合 Broadcast 规范

.. note::
    如想了解更多 Broadcasting 内容，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::
    - inputs (list(Tensor)|tuple(Tensor)) - 一组输入 Tensor，数据类型为：bool、float32、float64、int32 或 int64。所有的输入 Tensor 均需要满足 rank <= 5。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``list(Tensor)``，一组 Broadcast 后的 ``Tensor``，其顺序与 ``input`` 一一对应。

代码示例
:::::::::

COPY-FROM: paddle.broadcast_tensors
