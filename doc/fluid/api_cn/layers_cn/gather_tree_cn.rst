.. _cn_api_fluid_layers_gather_tree:

gather_tree
-------------------------------

.. py:function:: paddle.fluid.layers.gather_tree(ids, parents)

:alias_main: paddle.nn.gather_tree
:alias: paddle.nn.gather_tree,paddle.nn.decode.gather_tree
:old_api: paddle.fluid.layers.gather_tree



该OP在整个束搜索(Beam Search)结束后使用。在搜索结束后，可以获得每个时间步选择的的候选词id及其对应的在搜索树中的parent节点， ``ids`` 和 ``parents`` 的形状布局均为 :math:`[max\_time, batch\_size, beam\_size]` ，该OP从最后一个时间步回溯产生完整的id序列。


示例：

::

        给定:
            ids = [[[2 2]
                    [6 1]]
                   [[3 9]
                    [6 1]]
                   [[0 1]
                    [9 0]]]
            parents = [[[0 0]
                        [1 1]]
                       [[1 0]
                        [1 0]]
                       [[0 0]
                        [0 1]]]

        结果:                
            gather_tree(ids, parents)  
                        = [[[2 2]
                            [1 6]]
                           [[3 3]
                            [6 1]]
                           [[0 1]
                            [9 0]]]



参数：
    - **ids** (Variable) - 形状为 :math:`[length, batch\_size, beam\_size]` 的三维Tensor，数据类型是int32或int64。包含了所有时间步选择的id。
    - **parents** (Variable) - 形状和数据类型均与 ``ids`` 相同的Tensor。包含了束搜索中每一时间步所选id对应的parent。
    
返回：和 ``ids`` 具有相同形状和数据类型的Tensor。包含了根据parent回溯而收集产生的完整id序列。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    ids = fluid.data(name='ids',
                     shape=[5, 2, 2],
                     dtype='int64')
    parents = fluid.data(name='parents',
                         shape=[5, 2, 2],
                         dtype='int64')
    final_sequences = fluid.layers.gather_tree(ids, parents)





