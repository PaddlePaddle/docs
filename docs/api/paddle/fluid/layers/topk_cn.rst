.. _cn_api_fluid_layers_topk:

topk
-------------------------------
.. py:function:: paddle.fluid.layers.topk(input, k, name=None)




此 OP 用于查找输入 Tensor 的最后一维的前 k 个最大项，返回它们的值和索引。
如果输入是 1-D Tensor，则找到 Tensor 的前 k 个最大项，并输出前 k 个最大项的值和索引。如果输入是更高阶的 Tensor，则该 OP 会基于最后一维计算前 k 项。

- 例 1：

.. code-block:: python

    输入：
        input.shape = [3, 4]
        input.data = [[5, 4, 2, 3],
                     [9, 7, 10, 25],
                     [6, 2, 10, 1]]
        k = 2

    输出：
        第一个输出：
        values.shape = [3, 2]
        values.data = [[5, 4],
                      [10, 25],
                      [6, 10]]

        第二个输出：
        indices.shape = [3, 2]
        indices.data = [[0, 1],
                       [2, 3],
                       [0, 2]]


参数
::::::::::::

    - **input** (Variable) - 输入的 Tensor，支持的数据类型：float32，float64。
    - **k** (int|Variable) - 指定在输入 Tensor 最后一维中寻找最大前多少项。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - ``values``：输入 Tensor 最后维切片的最大前 k 项。数据类型同输入 Tensor 一致。Tensor 维度等于 :math:`input.shape[:-1]+ [k]` 。

    - ``indices``：输入 Tensor 最后维切片最大前 k 项值的索引，数据类型为 int64，维度同 values 的维度。

抛出异常
::::::::::::

    - ``ValueError``：如果 k<1 或者 k 大于输入的最后维。

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    input = layers.data(name="input", shape=[13, 11], dtype='float32')
    top5_values, top5_indices = layers.topk(input, k=5) #top5_values.shape=[13, 5]，top5_indices.shape=[13, 5]

    # 1D Tensor
    input1 = layers.data(name="input1", shape=[13], dtype='float32')
    top5_values, top5_indices = layers.topk(input1, k=5) #top5_values.shape=[5]，top5_indices.shape=[5]

    # k=Variable
    input2 = layers.data(name="input2", shape=[13, 11], dtype='float32')
    vk = layers.data(name="vk", shape=[1], dtype='int32') # 把 k 值保存在 vk.data[0]中
    vk_values, vk_indices = layers.topk(input2, k=vk) #vk_values.shape=[13, k]，vk_indices.shape=[13, k]
