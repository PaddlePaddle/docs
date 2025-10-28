.. _cn_api_paddle_geometric_reindex_heter_graph:

reindex_heter_graph
-------------------------------

.. py:function:: paddle.geometric.reindex_heter_graph(x, neighbors, count, value_buffer=None, index_buffer=None, name=None)

主要应用于图学习领域，需要与图采样相关的 API 配合使用，主要处理异构图场景。其主要目的是对输入的中心节点信息和邻居信息进行从 0 开始的重新编号，以方便后续的图模型子图训练。

.. note::
    输入 ``x`` 中的元素需保证是独有的，否则可能会带来一些潜在的错误。输入的节点将会和邻居节点一同从 0 进行编号。

以输入 x = [0, 1, 2] 作为例子解释。对于异构图 A ，假设我们有邻居 neighbors = [8, 9, 0, 4, 7, 6, 7]，以及邻居数量 count = [2, 3, 2]；
则可以得知节点 0 的邻居为 [8, 9]，节点 1 的邻居为 [0, 4, 7]，节点 2 的邻居为 [6, 7]。对于异构图 B，假设有邻居 neighbors = [0, 2, 3, 5, 1]，
以及邻居数量 count = [1, 3, 1]，则可以得知节点 0 的邻居为 [0]，节点 1 的邻居为 [2, 3, 5]。经过此 API 计算后，共计会返回三个结果：
    1. reindex_src: [3, 4, 0, 5, 6, 7, 6, 0, 2, 8, 9, 1]
    2. reindex_dst: [0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2]
    3. out_nodes: [0, 1, 2, 8, 9, 4, 7, 6, 3, 5]
可以看到 ``reindex_src`` 和 ``reindex_dst`` 中的值实际上是各个节点在 ``out_nodes`` 中对应的下标索引。

参数
:::::::::
    - **x** (Tensor) - 输入的中心节点原始编号，数据类型为：int32、int64。
    - **neighbors** (list | tuple) - 中心节点对应到各个异构图中的邻居节点编号，数据类型为：int32、int64。
    - **count** (list | tuple) - 中心节点对应到各个异构图中的邻居数目，数据类型为：int32。
    - **value_buffer** (Tensor，可选) - 用于快速哈希索引的缓存 Tensor，可加速重编号过程。数据类型为 int32，并且应当事先填充为-1。默认值为 None。
    - **index_buffer** (Tensor，可选) - 用于快速哈希索引的缓存 Tensor，可加速重编号过程。数据类型为 int32，并且应当事先填充为-1。默认值为 None。如果需要使用加速重编号过程，则 ``value_buffer`` 和 ``index_buffer`` 均不可为空。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - reindex_src (Tensor) - 重编号后的边对应的源节点信息。
    - reindex_dst (Tensor) - 重编号后的边对应的目标节点信息。
    - out_nodes (Tensor) - 返回去重后的输入中心节点信息和邻居信息，且为原始编号。注意，我们将输入的中心节点编号信息放置于前面，而邻居节点放置于后面。


代码示例
::::::::::

COPY-FROM: paddle.geometric.reindex_heter_graph
