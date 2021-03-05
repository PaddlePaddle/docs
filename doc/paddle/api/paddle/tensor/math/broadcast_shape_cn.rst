.. _cn_api_tensor_broadcast_shape:

broadcast_shape
-------------------------------

.. py:function:: paddle.broadcast_shape(x_shape, y_shape)


该函数返回对x_shape大小的张量和y_shape大小的张量做broadcast操作后得到的shape，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting` 。

参数
:::::::::
    - x_shape (list[int]|tuple[int]) - 输入Tensor的shape。
    - y_shape (list[int]|tuple[int]) - 输入Tensor的shape。

返回
:::::::::
broadcast操作后的shape，返回类型为 list[int]。


代码示例
:::::::::

..  code-block:: python

    import paddle

    shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
    # [2, 3, 3]
    
    # shape = paddle.broadcast_shape([2, 1, 3], [3, 3, 1])
    # ValueError (terminated with error message).

