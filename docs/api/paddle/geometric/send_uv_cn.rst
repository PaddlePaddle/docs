.. _cn_api_geometric_send_uv:

send_uv
-------------------------------

.. py:function:: paddle.geometric.send_uv(x, y, src_index, dst_index, message_op="add", name=None)

主要应用于图学习领域，目的是为了减少在消息传递过程中带来的中间变量显存或内存的损耗。其中，``x`` 作为输入的节点特征 Tensor，首先利用 ``src_index`` 作为索引来 gather 出在 ``x`` 中相应位置的数据，接着利用 ``dst_index`` gather 出 ``y`` 中相应位置的数据，再通过 ``message_op`` 确认计算方式，最终返回。其中，``message_op`` 包括另外 add、sub、mul、div 共计四种计算方式。

.. code-block:: text

        x = [[0, 2, 3],
             [1, 4, 5],
             [2, 6, 7]]

        src_index = [0, 1, 2, 0]

        dst_index = [1, 2, 1, 0]

        message_op = "add"

        Then:

        Out = [[0, 2, 3],
               [2, 8, 10],
               [1, 4, 5]]

参数
:::::::::
    - **x** (Tensor) - 输入的节点特征 Tensor，数据类型为：float32、float64、int32、int64。另外，我们在 GPU 计算中支持 float16。
    - **src_index** (Tensor) - 1-D Tensor，数据类型为：int32、int64。
    - **dst_index** (Tensor) - 1-D Tensor，数据类型为：int32、int64。注意：``dst_index`` 的形状应当与 ``src_index`` 一致。
    - **message_op** (str) - 不同计算方式，包括 add、sub、mul、div。默认值为 add。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor`` ，输出更新后的边特征。


代码示例
::::::::::

.. code-block:: python

     import paddle

     x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
     y = paddle.to_tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype="float32")
     indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
     src_index = indexes[:, 0]
     dst_index = indexes[:, 1]
     out = paddle.geometric.send_uv(x, y, src_index, dst_index, message_op="add")
     # Outputs: [[2., 5., 7.], [5., 9., 11.], [4., 9., 11.], [0., 3., 5.]]
