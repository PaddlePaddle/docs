.. _cn_api_paddle_geometric_weighted_sample_neighbors:

weighted_sample_neighbors
-------------------------------

.. py:function:: paddle.geometric.weighted_sample_neighbors(row, colptr, edge_weight, input_nodes, sample_size=- 1, eids=None, return_eids=False, name=None)

此 API 主要用于图学习领域，其主要目的是提供图加权采样方法的高性能。例如，我们获取输入图边缘的 CSC（Compressed Sparse Column，压缩列信息），分别对应 ``row`` 和 ``colptr``，从而将图转换为适用于邻居采样的格式，并且输入的 ``edge_weight`` 也应该与 CSC 格式匹配。

再输入需要进行采样的中心节点 ``input_nodes``，以及采样的邻居个数 ``sample_size``，从而可以获得对应中心节点采样后的邻居。

此 API 最终将返回加权采样的邻居数目，被选为邻居的概率与其权重相关，权重越高，被选概率越高。

参数
:::::::::
    - **row** (Tensor) - 输入原始图的 CSC 格式的行信息，数据类型为：int32、int64，形状为[num_edges, 1] 或 [num_edges]。
    - **colptr** (Tensor) - 输入原始图的 CSC 格式的压缩列信息，数据类型应当与 ``row`` 一致，形状为[num_nodes + 1, 1]或 [num_nodes + 1]。
    - **edge_weight** (Tensor) - 输入原始图边缘的 CSC 格式边权重。形状应为[num_edges, 1]或[num_edges]。可用的数据类型为 float32。
    - **input_nodes** (Tensor) - 需进行邻居采样的中心节点信息，数据类型应当与 ``row`` 一致。
    - **sample_size** (int) - 采样邻居个数。默认值为-1，表示采样输入中心节点的所有邻居。
    - **eids** (Tensor，可选) - 输入原始图在 CSC 格式下的边编号信息。如果 ``return_eids`` 为 True，则不能为空。数据类型应当与 ``row`` 一致。默认为 None，表示不需要返回边编号信息。
    - **return_eids** (bool) - 是否返回采样后对应的原始边编号信息，默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - out_neighbors (Tensor) - 返回采样后的邻居节点。
    - out_count (Tensor) - 返回中心节点各自对应的采样邻居数目，形状应该与 ``input_nodes`` 一致。
    - out_eids (Tensor) - 如果 ``return_eids`` 为 True，则会返回采样边对应的编号信息，否则不返回。

代码示例
::::::::::

COPY-FROM: paddle.geometric.weighted_sample_neighbors
