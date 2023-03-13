.. _cn_api_fluid_layers_lod_append:

lod_append
-------------------------------

.. py:function:: paddle.fluid.layers.lod_append(x, level)




给 ``x`` 的 LoD 添加 ``level`` 。

简单示例：

.. code-block:: python

    give a 1-level LodTensor x:
        x.lod = [[2, 3, 1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    level:[1, 1, 1, 1, 1, 1]

    Then we get a 2-level LodTensor:
        x.lod = [[2, 3, 1], [1, 1, 1, 1, 1, 1]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

参数
::::::::::::

    - **x** (Variable)-输入变量，可以是 LoDTensor 或 tensor。
    - **level** (list|tuple|Variable)-预添加到 x 的 LoD 里的 LoD level。

返回
::::::::::::
一个有着新的 LoD level 的输出变量

返回类型
::::::::::::
Variable

Raise: ``ValueError`` - 如果 y 为 None 或者 level 不可迭代。

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[6, 10], lod_level=1)
    out = fluid.layers.lod_append(x, [1,1,1,1,1,1])
