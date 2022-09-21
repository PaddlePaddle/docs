.. _cn_api_geometric_message_passing_send_u_recv:

send_u_recv
-------------------------------

.. py:function:: paddle.geometric.message_passing.send_u_recv(x, src_index, dst_index, reduce_op="sum", out_size=None, name=None)

主要应用于图学习领域，目的是为了减少在消息传递过程中带来的中间变量显存或内存的损耗，与 :ref:`cn_api_geometric_send_u_recv` 功能一致。
