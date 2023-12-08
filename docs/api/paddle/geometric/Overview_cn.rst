.. _cn_overview_paddle_geometric:

paddle.geometric
---------------------

paddle.geometric 目录下包含飞桨框架支持的图领域的相关 API。具体如下：

-  :ref:`高性能图消息传递 <faster_message_passing>`
-  :ref:`高效图采样 <faster_graph_sampling>`
-  :ref:`数学分段求值 <math_segment>`

.. _faster_message_passing:

高性能图消息传递
==========================

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.geometric.send_u_recv <cn_api_paddle_geometric_send_u_recv>` ", "节点特征消息传递"
    " :ref:`paddle.geometric.send_ue_recv <cn_api_paddle_geometric_send_ue_recv>` ", "节点融合边特征消息传递"
    " :ref:`paddle.geometric.send_uv <cn_api_paddle_geometric_send_uv>` ", "源节点与目标节点消息发送并计算"

.. _faster_graph_sampling:

高效图采样
==========================

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.geometric.sample_neighbors <cn_api_paddle_geometric_sample_neighbors>` ", "无权重邻居采样"
    " :ref:`paddle.geometric.weighted_sample_neighbors <cn_api_paddle_geometric_weighted_sample_neighbors>` ", "加权近邻采样"
    " :ref:`paddle.geometric.reindex_graph <cn_api_paddle_geometric_reindex_graph>` ", "同构图场景下的子图重编号"
    " :ref:`paddle.geometric.reindex_heter_graph <cn_api_paddle_geometric_reindex_heter_graph>` ", "异构图场景下的子图重编号"

.. _math_segment:

数学分段求值
==========================

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.geometric.segment_sum <cn_api_paddle_geometric_segment_sum>` ", "分段求和"
    " :ref:`paddle.geometric.segment_mean <cn_api_paddle_geometric_segment_mean>` ", "分段求均值"
    " :ref:`paddle.geometric.segment_max <cn_api_paddle_geometric_segment_max>` ", "分段求最大值"
    " :ref:`paddle.geometric.segment_min <cn_api_paddle_geometric_segment_min>` ", "分段求最小值"
