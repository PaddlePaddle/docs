.. _cn_api_fluid_layers_scatter:

scatter
-------------------------------

.. py:function:: paddle.fluid.layers.scatter(input, index, updates, name=None, overwrite=True)


通过更新输入在第一维度上指定索引位置处的元素来获得输出。

.. math::
          \\Out=XOut[Ids]=Updates\\


参数：
  - **input** （Variable） - 秩> = 1的源输入。
  - **index** （Variable） - 秩= 1的索引输入。 它的dtype应该是int32或int64，因为它用作索引。
  - **updates** （Variable） - scatter 要进行更新的变量。
  - **name** （str | None） - 输出变量名称。 默认None。
  - **overwrite** （bool） - 具有相同索引时更新输出的模式。如果为True，则使用覆盖模式更新相同索引的输出，如果为False，则使用accumulate模式更新相同索引的grad。默认值为True。您可以设置overwrite=False以使用scatter_add。

返回：张量变量, 与输入张量的shape相同

返回类型：output（Variable）

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
     
    input = fluid.layers.data(name='data', shape=[3, 5, 9], dtype='float32', append_batch_size=False)
    index = fluid.layers.data(name='index', shape=[3], dtype='int64', append_batch_size=False)
    updates = fluid.layers.data(name='update', shape=[3, 5, 9], dtype='float32', append_batch_size=False)
    
    output = fluid.layers.scatter(input, index, updates)











