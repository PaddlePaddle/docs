.. _cn_api_fluid_layers_gather:

gather
-------------------------------

.. py:function:: paddle.fluid.layers.gather(input, index, overwrite=True)

收集层（gather layer）

根据索引index获取X的最外层维度的条目，并将它们串连在一起。

.. math::
                        Out=X[Index]

::

        X = [[1, 2],
             [3, 4],
             [5, 6]]

        Index = [1, 2]

        Then:

        Out = [[3, 4],
               [5, 6]]


参数:
         - **input** (Variable) - input的秩rank >= 1。
        - **index** (Variable) - index的秩rank = 1。
        - **overwrite** (bool) - 具有相同索引时更新grad的模式。如果为True，则使用覆盖模式更新相同索引的grad，如果为False，则使用accumulate模式更新相同索引的grad。Default值为True。

返回：和输入的秩相同的输出张量。

返回类型：output (Variable)

**代码示例**

..  code-block:: python
  
  import paddle.fluid as fluid
  x = fluid.layers.data(name='x', shape=[-1, 5], dtype='float32')
  index = fluid.layers.data(name='index', shape=[-1, 1], dtype='int32')
  output = fluid.layers.gather(x, index)









