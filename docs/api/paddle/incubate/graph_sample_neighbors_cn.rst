.. _cn_api_incubate_graph_sample_neighbors:

graph_sample_neighbors
-------------------------------

.. py:function:: paddle.incubate.graph_sample_neighbors(row, colptr, input_nodes, eids=None, perm_buffer=None, sample_size=-1, return_eids=False, flag_perm_buffer=False, name=None)

主要应用于图学习领域，主要目的是提供高性能图邻居采样方法。通过输入图的CSC（Compressed Sparse Column，压缩列信息），分别对应 ``row`` 和 ``colptr`` ，从而将图转换为适用于邻居采样的格式，再输入需要进行采样的中心节点 ``input_nodes`` ，以及采样的邻居个数 ``sample_size`` ，从而可以获得对应中心节点采样后的邻居。另外，在GPU版本提供了Fisher-yates高性能图采样方法。

参数
:::::::::
    - row (Tensor) - 输入原始图的CSC格式的行信息，数据类型为：int32、int64，形状为[num_edges, 1] 或 [num_edges]。
    - colptr (Tensor) - 输入原始图的CSC格式的压缩列信息，数据类型应当与 ``row`` 一致，形状为[num_nodes + 1, 1]或 [num_nodes + 1]。
    - input_nodes (Tensor) - 需进行邻居采样的中心节点信息，数据类型应当与 ``row`` 一致。
    - eids (Tensor，可选) - 输入原始图在CSC格式下的边编号信息。如果 ``return_eids`` 为True，则不能为空。数据类型应当与 ``row`` 一致。默认为None，表示不需要返回边编号信息。
    - perm_buffer (Tensor，可选) - Fisher-yates采样方法需要用到的缓存Tensor。如果 ``flag_perm_buffer`` 为True，则不能为空。数据类型应当与 ``row`` 一致，形状为[num_edges]，填充内容为0 至 num_edges的顺序递增序列。
    - sample_size (int) - 采样邻居个数。默认值为-1，表示采样输入中心节点的所有邻居。
    - return_eids (bool) - 是否返回采样后对应的原始边编号信息，默认为False。
    - flag_perm_buffer (bool) - 是否采用Fisher-yates采样方法，默认为False。 
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
    - out_neighbors (Tensor) - 返回采样后的邻居节点。
    - out_count (Tensor) - 返回中心节点各自对应的采样邻居数目，形状应该与 ``input_nodes`` 一致。
    - out_eids (Tensor) - 如果 ``return_eids`` 为True，则会返回采样边对应的编号信息，否则不返回。


代码示例
::::::::::

.. code-block:: python

    import paddle
 
    # edges: (3, 0), (7, 0), (0, 1), (9, 1), (1, 2), (4, 3), (2, 4),
    #        (9, 5), (3, 5), (9, 6), (1, 6), (9, 8), (7, 8)
    row = [3, 7, 0, 9, 1, 4, 2, 9, 3, 9, 1, 9, 7]
    colptr = [0, 2, 4, 5, 6, 7, 9, 11, 11, 13, 13]
    nodes = [0, 8, 1, 2]
    sample_size = 2
    row = paddle.to_tensor(row, dtype="int64")
    colptr = paddle.to_tensor(colptr, dtype="int64")
    nodes = paddle.to_tensor(nodes, dtype="int64")
    out_neighbors, out_count = \
        paddle.incubate.graph_sample_neighbors(row, colptr, nodes, 
                                               sample_size=sample_size)
