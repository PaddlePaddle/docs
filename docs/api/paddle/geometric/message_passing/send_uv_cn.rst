.. _cn_api_geometric_message_passing_send_uv:

send_uv
-------------------------------

.. py:function:: paddle.geometric.message_passing.send_uv(x, y, src_index, dst_index, message_op="add", name=None)

主要应用于图学习领域，目的是为了减少在消息传递过程中带来的中间变量显存或内存的损耗，与 :ref:`cn_api_geometric_send_uv` 功能一致。
