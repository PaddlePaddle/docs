.. _cn_api_fluid_layers_scatter_nd_add:

scatter_nd_add
-------------------------------

.. py:function:: paddle.fluid.layers.scatter_nd_add(ref, index, updates, name=None)

该OP通过对Variable中的单个值或切片应用稀疏加法，从而得到输出的Variable。 :code:`ref` 是维度为 :code:`R` 的张量，:code:`index` 是维度为 :code:`K` 的张量。因此， :code:`index` 的形状是 :math:`[i_0, i_1, ..., i_{K-2}, Q]` ，其中  :math:`Q \leq R` 。:code:`updates` 是一个维度为 :math:`K - 1 + R - Q` 的张量，它的形状是 :math:`index.shape[:-1] + ref.shape[index.shape[-1]:]` 。根据 :code:`index` 的 :math:`[i_0, i_1, ..., i_{K-2}]` 得到相应的 :code:`updates` 切片，将其加到根据 :code:`index` 的最后一维得到 :code:`ref` 切片上，从而得到最终的输出张量。  


.. code-block:: python

        - 案例 1:
            ref = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          得到:
             
            output = [0, 22, 12, 14, 4, 5]

        - 案例 2:
            ref = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            ref.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          得到:
             
            output = [[67, 19], [-16, -27]]


参数：
    - **ref** (Variable) - 输入张量，数据类型可以是int32，int64，float32，float64。
    - **index** (Variable) - 输入的索引张量，数据类型为非负int32或非负int64。它的维度 :code:`index.rank` 必须大于1，并且 :code:`index.shape[-1] <= ref.rank`
    - **updates** (Variable) - 输入的更新张量，它必须和 :code:`ref` 有相同的数据类型。形状必须是 :code:`index.shape[:-1] + ref.shape[index.shape[-1]:]` 。
    - **name** (string) - 该层的名字，默认值为None，表示会自动命名。
    
返回：数据类型和形状都与 :code:`ref` 相同的Tensor|LoDTensor。

返回类型：Variable

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        ref = fluid.layers.data(name='ref', shape=[3, 5, 9, 10], dtype='float32', append_batch_size=False)
        index = fluid.layers.data(name='index', shape=[3, 2], dtype='int32', append_batch_size=False)
        updates = fluid.layers.data(name='update', shape=[3, 9, 10], dtype='float32', append_batch_size=False)
        output = fluid.layers.scatter_nd_add(ref, index, updates)



