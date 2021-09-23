.. _cn_api_paddle_broadcast_tensors:

broadcast_tensors
-------------------------------

.. py:function:: paddle.broadcast_tensors(inputs, name=None)

根据Broadcast规范对一组输入 ``inputs`` 进行Broadcast操作
输入应符合Broadcast规范

.. note::
    如您想了解更多Broadcasting内容，请参见 :ref:`cn_user_guide_broadcasting` 。

参数
:::::::::
    - inputs (list(Tensor)|tuple(Tensor)) - 一组输入Tensor，数据类型为：bool、float32、float64、int32或int64。
                                          - 所有的输入Tensor均需要满足rank <= 5
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``list(Tensor)`` ，一组Broadcast后的 ``Tensor``，其顺序与 ``input`` 一一对应。

代码示例
:::::::::

.. code-block:: python

    import paddle
    
    x1 = paddle.rand([1, 2, 3, 4]).astype('float32')
    x2 = paddle.rand([1, 2, 1, 4]).astype('float32')
    x3 = paddle.rand([1, 1, 3, 1]).astype('float32')

    out1, out2, out3 = paddle.broadcast_tensors(input=[x1, x2, x3])
    # out1, out2, out3: 分别对应x1, x2, x3 Broadcast的结果，其形状均为 [1,2,3,4]
