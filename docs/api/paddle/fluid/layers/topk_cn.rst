.. _cn_api_fluid_layers_topk:

topk
-------------------------------
.. py:function:: paddle.fluid.layers.topk(input, k, name=None)




此OP用于查找输入Tensor的最后一维的前k个最大项，返回它们的值和索引。
如果输入是1-D Tensor，则找到Tensor的前k个最大项，并输出前k个最大项的值和索引。如果输入是更高阶的Tensor，则该OP会基于最后一维计算前k项。

- 例1：

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

    - **input** (Variable) - 输入的Tensor，支持的数据类型：float32，float64。
    - **k** (int|Variable) - 指定在输入Tensor最后一维中寻找最大前多少项。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - ``values``：输入Tensor最后维切片的最大前k项。数据类型同输入Tensor一致。Tensor维度等于 :math:`input.shape[:-1]+ [k]` 。

    - ``indices``：输入Tensor最后维切片最大前k项值的索引，数据类型为int64，维度同values的维度。

抛出异常
::::::::::::

    - ``ValueError``：如果k<1或者k大于输入的最后维。

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
    vk = layers.data(name="vk", shape=[1], dtype='int32') # 把k值保存在vk.data[0]中
    vk_values, vk_indices = layers.topk(input2, k=vk) #vk_values.shape=[13, k]，vk_indices.shape=[13, k]








