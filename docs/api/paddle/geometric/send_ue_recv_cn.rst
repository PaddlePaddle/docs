.. _cn_api_geometric_send_ue_recv:

send_ue_recv
-------------------------------

.. py:function:: paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum", out_size=None, name=None)

主要应用于图学习领域，目的是为了减少在消息传递过程中带来的中间变量显存或内存的损耗。其中，``x`` 作为输入节点特征 Tensor，首先利用 ``src_index`` 作为索引来 gather 出在 ``x`` 中相应位置的数据，接着与边特征 Tensor ``e`` 进行计算，计算方式包括 add、sub、mul、div。随后再将计算的结果利用 ``dst_index`` 来更新到对应的输出结果中，其中 ``message_op`` 表示输入 ``x`` 和 ``e`` 之间的计算方式， ``reduce_op`` 表示不同的结果更新方式，包括 sum、mean、max、min 共计 4 种处理模式。另外，提供了 ``out_size`` 参数，用于设置实际输出的形状，有利于减少实际显存占用。

.. code-block:: text

        x = [[0, 2, 3],
             [1, 4, 5],
             [2, 6, 7]]

        y = [1, 1, 1]

        src_index = [0, 1, 2, 0]

        dst_index = [1, 2, 1, 0]

        message_op = "add"

        reduce_op = "sum"

        out_size = None

        Then:

        Out = [[1, 3, 4],
               [4, 10, 12],
               [2, 5, 6]]

参数
:::::::::
    - **x** (Tensor) - 输入的节点特征 Tensor，数据类型为：float32、float64、int32、int64。另外，我们在 GPU 计算中支持 float16。
    - **y** (Tensor) - 输入的边特征 Tensor，数据类型为：float32、float64、int32、int64。数据类型需与 ``x`` 相同。另外，我们在 GPU 计算中支持 float16。
    - **src_index** (Tensor) - 1-D Tensor，数据类型为：int32、int64。
    - **dst_index** (Tensor) - 1-D Tensor，数据类型为：int32、int64。注意：``dst_index`` 的形状应当与 ``src_index`` 一致。
    - **message_op** (str) - 不同计算方式，包括 add、sub、mul、div。默认值为 add。
    - **reduce_op** (str) - 不同更新方式，包括 sum、mean、max、min。默认值为 sum。
    - **out_size** (int64 | Tensor | None) - 可以通过根据实际需求设置 ``out_size`` 来改变实际输出形状。默认值为 None，表示这个参数将不会被使用。注意，``out_size`` 的值必须等于或大于 ``max(dst_index) + 1`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor`` ，维度和数据类型都与 ``x`` 相同，存储运算后的结果。如果 ``out_size`` 参数正确设置了，则输出结果的第 0 维大小是 ``out_size`` ，其余维度大小与 ``x`` 相同。


代码示例
::::::::::

COPY-FROM: paddle.geometric.send_ue_recv
