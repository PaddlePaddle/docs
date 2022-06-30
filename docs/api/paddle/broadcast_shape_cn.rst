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

COPY-FROM: paddle.broadcast_shape