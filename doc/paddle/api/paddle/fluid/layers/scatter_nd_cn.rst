.. _cn_api_fluid_layers_scatter_nd:

scatter_nd
-------------------------------

.. py:function:: paddle.scatter_nd(index, updates, shape, name=None)




该OP根据 :code:`index` ，将 :code:`updates` 添加到一个新的张量中，从而得到输出的Tensor。这个操作与 :code:`scatter_nd_add` 类似，除了形状为 :code:`shape` 的张量是通过零初始化的。相应地， :code:`scatter_nd(index, updates, shape)` 等价于 :code:`scatter_nd_add(fluid.layers.zeros(shape, updates.dtype), index, updates)` 。如果 :code:`index` 有重复元素，则将累积相应的更新，因此，由于数值近似问题，索引中重复元素的顺序不同可能会导致不同的输出结果。具体的计算方法可以参见 :code:`scatter_nd_add` 。该OP是 :code:`gather_nd` 的反函数。

参数：
    - **index** (Tensor) - 输入的索引张量，数据类型为非负int32或非负int64。它的维度 :code:`index.ndim` 必须大于1，并且 :code:`index.shape[-1] <= len(shape)`
    - **updates** (Tensor) - 输入的更新张量。形状必须是 :code:`index.shape[:-1] + shape[index.shape[-1]:]` 。数据类型可以是float32，float64。
    - **shape** (tuple|list) - 要求输出张量的形状。类型是tuple或者list。
    - **name** (string) - 该层的名字，默认值为None，表示会自动命名。
    
返回：数据类型与 :code:`updates` 相同，形状是 :code:`shape` 的Tensor。

返回类型：Tensor

**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np
    index_data = np.array([[1, 1],
                            [0, 1],
                            [1, 3]]).astype(np.int64)
    index = paddle.to_tensor(index_data)
    updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
    shape = [3, 5, 9, 10]
    output = paddle.scatter_nd(index, updates, shape)
        


