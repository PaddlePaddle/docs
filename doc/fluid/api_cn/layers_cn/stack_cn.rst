.. _cn_api_fluid_layers_stack:

stack
-------------------------------

.. py:function:: paddle.fluid.layers.stack(x, axis=0)

实现了stack层。

沿 ``axis`` 轴，该层对输入 ``x`` 进行stack运算。

输入 x 可以是单个变量, 或是多个变量组成的列表或元组。如果 x 是一个列表或元组, 那么这些变量必须同形。 假设每个输入的形都为 :math:`[d_0,d_1,...,d_{n−1}]` , 则输出变量的形为 :math:`[d_0,d_1,...,d_{axis}=len(x),...,d_{n−1}]` 。 如果 ``axis`` < 0, 则将其取代为 :math:`axis+rank(x[0])+1` 。 如果 ``axis`` 为 None, 则认为它是 0。


例如：

.. code-block:: text

    例1:
      输入:
        x[0].data = [ [1.0 , 2.0 ] ]
        x[0].dims = [1, 2]
        x[1].data = [ [3.0 , 4.0 ] ]
        x[1].dims = [1, 2]
        x[2].data = [ [5.0 , 6.0 ] ]
        x[2].dims = [1, 2]

      参数:
        axis = 0

      输出:
        Out.data =[ [ [1.0, 2.0] ],
                    [ [3.0, 4.0] ],
                    [ [5.0, 6.0] ] ]
        Out.dims = [3, 1, 2]

    例2:
      如果
        x[0].data = [ [1.0 , 2.0 ] ]
        x[0].dims = [1, 2]
        x[1].data = [ [3.0 , 4.0 ] ]
        x[1].dims = [1, 2]
        x[2].data = [ [5.0 , 6.0 ] ]
        x[2].dims = [1, 2]

      参数:
        axis = 1 or axis = -2

      输出:
        Out.data =[ [ [1.0, 2.0]
                      [3.0, 4.0]
                      [5.0, 6.0] ] ]
        Out.dims = [1, 3, 2]

参数:

  - **x** (Variable|list(Variable)|tuple(Variable)) – 输入变量
  - **axis** (int|None) – 对输入进行stack运算所在的轴

返回: 经stack运算后的变量

返回类型: Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    x1 = layers.data(name='x1', shape=[1, 2], dtype='int32')
    x2 = layers.data(name='x2', shape=[1, 2], dtype='int32')
    data = layers.stack([x1,x2])







