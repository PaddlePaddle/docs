.. _cn_api_fluid_layers_topk:

topk
-------------------------------
.. py:function:: paddle.fluid.layers.topk(input, k, name=None)

这个算子用于查找最后一维的前k个最大项，返回它们的值和索引。

如果输入是（1-D Tensor），则找到向量的前k最大项，并以向量的形式输出前k最大项的值和索引。values[j]是输入中第j最大项，其索引为indices[j]。
如果输入是更高阶的张量，则该operator会基于最后一维计算前k项

例如：

.. code-block:: text


    如果:
        input = [[5, 4, 2, 3],
                [9, 7, 10, 25],
                [6, 2, 10, 1]]
        k = 2

    则:
        第一个输出:
        values = [[5, 4],
                [10, 25],
                [6, 10]]

        第二个输出:
        indices = [[0, 1],
                [2, 3],
                [0, 2]]

参数：
    - **input** (Variable)-输入变量可以是一个向量或者更高阶的张量
    - **k** (int|Variable)-在输入最后一维中寻找的前项数目
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名。默认为空

返回：含有两个元素的元组。元素都是变量。第一个元素是最后维切片的前k项。第二个元素是输入最后维里值索引

返回类型：元组[变量]

抛出异常: ``ValueError`` - 如果k<1或者k不小于输入的最后维

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    input = layers.data(name="input", shape=[13, 11], dtype='float32')
    top5_values, top5_indices = fluid.layers.topk(input, k=5)









