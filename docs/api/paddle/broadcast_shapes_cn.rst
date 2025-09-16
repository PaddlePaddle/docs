.. _cn_api_paddle_broadcast_shapes:

broadcast_shapes
-------------------------------

.. py:function:: paddle.broadcast_shapes(*shapes: Sequence[int])

计算多个 Tensor shape 经过广播（broadcasting）之后的结果 shape。


参数
::::::::::::

    - **shapes** (Sequence[int]) - 一个或多个 Tensor 的 shape。每个 shape 都是一个整数序列（List 或 Tuple）。

返回
::::::::::::
list[int]，广播后的结果 shape。


代码示例
::::::::::::

COPY-FROM: paddle.broadcast_shapes
