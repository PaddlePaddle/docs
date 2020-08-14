.. _cn_api_tensor_tile: 
tile
-------------------------------

.. py:function:: paddle.tile(x, repeat_times, name=None)

:alias_main: paddle.tile
:alias: paddle.tile,paddle.tensor.tile,paddle.tensor.manipulation.tile


该OP会根据参数 ``repeat_times`` 对输入 ``x`` 的各维度进行复制。``x`` 的秩应小于等于6，并且repeat_times中的元素数量应该小于等于6。假设参数 ``repeat_times`` 中的元素数量为 ``d`` 且输入 ``x`` 的秩为 ``r`` ，那么输出张量的秩为 ``max(d,r)`` 。如果 ``d < r`` ，那么首先扩展 ``repeat_times`` 中的元素为 ``r`` ，扩展方式是在前面插入一个或单个元素 ``1`` 。例如，假设输入 ``x`` 的形状为 ``(3,)`` ，且 ``d`` 的值为2，那么输入 ``x`` 首先被扩展为形状为 ``(1,3)`` 的2-维张量。反之，如果 ``d > r`` ，那么首先将输入 ``x`` 扩展为d-维张量，扩展方式是为输入增加一个或多个维度值为1的高纬度。例如，假设输入 ``x`` 的形状为 ``(4,3,2,2)`` ，且 ``repeat_times`` 为元组 ``(3,2)`` ，那么输入 ``repeat_times`` 首先被扩展为元组为 ``(1,1,3,2)`` 。以下是一个用例：

::

        输入(x) 是一个形状为[2, 3, 1]的 3-D Tensor :

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        属性(repeat_times):  [1, 2, 2]

        输出(out) 是一个形状为[2, 6, 2]的 3-D Tensor:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]

参数
:::::::::
        - x (Tensor) - 输入的Tensor，数据类型为：float32、float64、int32、bool。
        - repeat_times (list|tuple|Variable) - 指定输入 ``x`` 的每个维度的复制次数，数据类型为：int32。如果 ``repeat_times`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为(1,)的张量。如果 ``repeat_times`` 的类型是张量，则是1-维张量。
        - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor`` ，数据类型与 ``x`` 相同。返回值的每个维度的大小等于 ``x`` 的相应维度的大小乘以 ``repeat_times`` 给出的相应值。
..
  返回类型：``Tensor`` 。

抛出异常
:::::::::
    - :code:`TypeError`：``repeat_times`` 的类型应该是 list、tuple 或 Variable。
    - :code:`ValueError`：``repeat_times`` 中的元素不能是负值。



