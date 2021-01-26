.. _cn_api_fluid_layers_gather_tree:

gather_tree
-------------------------------

.. py:function:: paddle.nn.functional.gather_tree(ids, parents)




该OP在整个束搜索(Beam Search)结束后使用。在搜索结束后，可以获得每个时间步选择的的候选词 id 及其对应的在搜索树中的 parent 节点， ``ids`` 和 ``parents`` 的形状布局均为 :math:`[max\_time, batch\_size, beam\_size]` ，该OP从最后一个时间步回溯产生完整的 id 序列。


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
    - **ids** (Tensor) - 形状为 :math:`[length, batch\_size, beam\_size]` 的三维 Tensor，数据类型是 int32 或 int64。包含了所有时间步选择的 id。
    - **parents** (Tensor) - 形状和数据类型均与 ``ids`` 相同的 Tensor。包含了束搜索中每一时间步所选 id 对应的 parent。
    
返回：和 ``ids`` 具有相同形状和数据类型的 Tensor。包含了根据 parent 回溯而收集产生的完整 id 序列。

返回类型：Tensor

**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np

    ids = paddle.to_tensor(np.asarray([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]]))

    parents = paddle.to_tensor(np.asarray([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]]))
    
    # [[[2, 2], [1, 6]], [[3, 3], [6, 1]], [[0, 1], [9, 0]]]
    final_sequences = paddle.nn.functional.gather_tree(ids, parents)





