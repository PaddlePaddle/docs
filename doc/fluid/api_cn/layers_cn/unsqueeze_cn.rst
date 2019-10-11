.. _cn_api_fluid_layers_unsqueeze:

unsqueeze
-------------------------------

.. py:function:: paddle.fluid.layers.unsqueeze(input, axes, name=None)

该OP向输入（input）的shape中一个或多个位置（axes）插入维度

- 示例：

.. code-block:: python

    输入：
      X.shape = [2, 3]
      X.data = [[1, 2, 3], 
                [4，5，6]]
      axes = [0, 2]
    输出（在X的第0维和第2维插入新维度）：
      Out.shape = [1, 2, 1, 3]
      Out.data = [[[[1, 2, 3]],
                    [[4, 5, 6]]]]
      
参数：
    - **input** (Variable) - 维度为 :math:[N_1, N2, ..., N_D]的多维Tensor
    - **axes** (list)- 整数数列，每个数代表要插入维度的位置
    - **name** (str|None) - 该层名称

返回：扩展维度后的多维Tensor

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[5, 10])
    y = fluid.layers.unsqueeze(input=x, axes=[1])
