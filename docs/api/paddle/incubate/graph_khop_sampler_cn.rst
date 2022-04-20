.. _cn_api_incubate_graph_khop_sampler:

graph_khop_sampler
-------------------------------

.. py:function:: paddle.incubate.graph_khop_sampler(row, colptr, input_nodes, sample_sizes, sorted_eids=None, return_eids=False, name=None)

此API主要应用于图学习领域，将节点邻居采样和节点重编号两步骤统一在一起，同时提供多层邻居采样，方便用户使用。关于邻居采样和节点重编号的相关API可以分别参考 :ref: `cn_api_incubate_graph_sample_neighbors` 和 :ref: `cp_api_incubate_graph_reindex` 。

参数
:::::::::
    - row (Tensor) - 输入原始图的CSC格式的行信息，数据类型为：int32、int64，形状为[num_edges, 1] 或 [num_edges]。
    - colptr (Tensor) - 输入原始图的CSC格式的压缩列信息，数据类型应当与 ``row`` 一致，形状为[num_nodes + 1, 1]或 [num_nodes + 1]。
    - input_nodes (Tensor) - 需进行邻居采样的中心节点信息，数据类型应当与 ``row`` 一致。
    - sample_sizes (list | tuple) - 表示我们需要每一层需要采样的邻居个数，数据类型为int。
    - sorted_eids (Tensor) - 输入原始图在CSC格式下的节点编号信息。如果 ``return_eids`` 为True，则不能为空。数据类型应当与 ``row`` 一致。
    - return_eids (bool) - 是否返回采样后对应的原始边编号信息，默认为False。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
    - edge_src (Tensor) - 返回采样后重索引的边对应的源节点信息。
    - edge_dst (Tensor) - 返回采样后重索引的边对应的目标节点信息。
    - sample_index (Tensor) - 返回去重后的输入中心节点信息和邻居信息，且为原始编号。
    - reindex_nodes (Tensor) - 返回输入中心节点在 ``sample_index`` 中的下标索引位置。
    - edge_eids (Tensor) - 如果 ``return_eids`` 为True，则会返回采样边对应的编号信息，否则不返回。


代码示例
::::::::::

.. code-block:: python

    import paddle

    row = [3, 7, 0, 9, 1, 4, 2, 9, 3, 9, 1, 9, 7]
    colptr = [0, 2, 4, 5, 6, 7, 9, 11, 11, 13, 13]
    nodes = [0, 8, 1, 2]
    sample_sizes = [2, 2]
    row = paddle.to_tensor(row, dtype="int64")
    colptr = paddle.to_tensor(colptr, dtype="int64")
    nodes = paddle.to_tensor(nodes, dtype="int64")

    edge_src, edge_dst, sample_index, reindex_nodes = \
        paddle.incubate.graph_khop_sampler(row, colptr, nodes, sample_sizes, False)

