.. _cn_api_incubate_graph_khop_sampler:

graph_khop_sampler
-------------------------------

.. py:function:: paddle.incubate.graph_khop_sampler(row, colptr, input_nodes, sample_sizes, sorted_eids=None, return_eids=False, name=None)

主要应用于图学习领域，将节点邻居采样和节点重编号两步骤统一在一起，同时提供多层邻居采样的功能。关于邻居采样和节点重编号的相关 API 可以分别参考 :ref:`cn_api_incubate_graph_sample_neighbors` 和 :ref:`cn_api_incubate_graph_reindex` 。

参数
:::::::::
    - **row** (Tensor) - 输入原始图的 CSC 格式的行信息，数据类型为：int32、int64，形状为[num_edges, 1] 或 [num_edges]。
    - **colptr** (Tensor) - 输入原始图的 CSC 格式的压缩列信息，数据类型应当与 ``row`` 一致，形状为[num_nodes + 1, 1]或 [num_nodes + 1]。
    - **input_nodes** (Tensor) - 需进行邻居采样的中心节点信息，数据类型应当与 ``row`` 一致。
    - **sample_sizes** (list|tuple) - 表示每一层需要采样的邻居个数，数据类型为 int。
    - **sorted_eids** (Tensor，可选) - 输入原始图在 CSC 格式下的边编号信息。如果 ``return_eids`` 为 True，则不能为空。数据类型应当与 ``row`` 一致。默认值为 None，表示不需要返回边编号信息。
    - **return_eids** (bool) - 是否返回采样后对应的原始边编号信息，默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - edge_src (Tensor) - 返回采样后重索引的边对应的源节点信息。
    - edge_dst (Tensor) - 返回采样后重索引的边对应的目标节点信息。
    - sample_index (Tensor) - 返回去重后的输入中心节点信息和邻居信息，且为原始编号。
    - reindex_nodes (Tensor) - 返回输入中心节点在 ``sample_index`` 中的下标索引位置。
    - edge_eids (Tensor) - 如果 ``return_eids`` 为 True，则会返回采样边对应的编号信息，否则不返回。


代码示例
::::::::::

COPY-FROM: paddle.incubate.graph_khop_sampler
