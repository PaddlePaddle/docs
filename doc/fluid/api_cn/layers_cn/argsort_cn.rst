.. _cn_api_fluid_layers_argsort:

argsort
-------------------------------

.. py:function:: paddle.fluid.layers.argsort(input,axis=-1,name=None)

对输入变量沿给定轴进行排序，输出排序好的数据和相应的索引，其维度和输入相同

.. code-block:: text

    例如：
  给定 input 并指定 axis=-1

        input = [[0.15849551, 0.45865775, 0.8563702 ],
                [0.12070083, 0.28766365, 0.18776911]],

      执行argsort操作后，得到排序数据：

        out = [[0.15849551, 0.45865775, 0.8563702 ],
            [0.12070083, 0.18776911, 0.28766365]],

  根据指定axis排序后的数据indices变为:

        indices = [[0, 1, 2],
                [0, 2, 1]]

参数：
    - **input** (Variable)-用于排序的输入变量
    - **axis** (int)- 沿该参数指定的轴对输入进行排序。当axis<0,实际的轴为axis+rank(input)。默认为-1，即最后一维。
    - **name** (str|None)-（可选）该层名称。如果设为空，则自动为该层命名。

返回：一组已排序的数据变量和索引

返回类型：元组

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
    out, indices = fluid.layers.argsort(input=x, axis=0)









