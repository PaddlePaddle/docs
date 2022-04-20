.. _cn_api_incubate_graph_reindex:

graph_reindex
-------------------------------

.. py:function:: paddle.incubate.graph_reindex(x, neighbors, count, value_buffer=None, index_buffer=None, flag_buffer_hashtable=False, name=None)

此API主要应用于图学习领域，需要与图采样相关的API配合使用。其主要目的是对输入的中心节点信息和邻居信息进行从0开始的重新编号，以方便后续的图模型子图训练。

.. code-block:: text

        X = [0, 1, 2]

        neighbors = [8, 9, 0, 4, 7, 6, 7]

        count = [2, 3, 2]

        value_buffer = None

        index_buffer = None

        flag_buffer_hashtable = False

        Then:

        reindex_src = [3, 4, 0, 5, 6, 7, 6]

        reindex_dst = [0, 0, 1, 1, 1, 2, 2]

        out_nodes = [0, 1, 2, 8, 9, 4, 7, 6]  # 可以将对应位置的节点编号替换到重编号的边中，得到重编号前的边信息。

参数
:::::::::
    - x (Tensor) - 输入的中心节点原始编号，数据类型为：int32、int64。
    - neighbors (Tensor) - 中心节点的邻居节点编号，数据类型为：int32、int64。
    - count (Tensor) - 中心节点各自的邻居数目，数据类型为：int32。
    - value_buffer (Tensor | None) - 用于快速哈希索引的缓存Tensor，可加速重编号过程。数据类型为int32，并且应当事先填充为-1。
    - index_buffer (Tensor | None) - 用于快速哈希索引的缓存Tensor，可加速重编号过程。数据类型为int32，并且应当事先填充为-1。
    - flag_buffer_hashtable (bool) - 是否采取快速哈希索引，默认为False。只适用于GPU版本的API。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
    - reindex_src (Tensor) - 重编号后的边对应的源节点信息。
    - reindex_dst (Tensor) - 重编号后的边对应的目标节点信息。
    - out_nodes (Tensor) - 返回去重后的输入中心节点信息和邻居信息，且为原始编号。注意，我们将输入的中心节点编号信息放置于前面，而邻居信息放置于后面。


代码示例
::::::::::

.. code-block:: python

    import paddle

    x = [0, 1, 2]
    neighbors = [8, 9, 0, 4, 7, 6, 7]
    count = [2, 3, 2]
    x = paddle.to_tensor(x, dtype="int64")
    neighbors = paddle.to_tensor(neighbors, dtype="int64")
    count = paddle.to_tensor(count, dtype="int32")
    
    reindex_src, reindex_dst, out_nodes = \
        paddle.incubate.graph_reindex(x, neighbors, count)
    # reindex_src: [3, 4, 0, 5, 6, 7, 6]
    # reindex_dst: [0, 0, 1, 1, 1, 2, 2]
    # out_nodes: [0, 1, 2, 8, 9, 4, 7, 6]
