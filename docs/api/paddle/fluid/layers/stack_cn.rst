.. _cn_api_fluid_layers_stack:

stack
-------------------------------

.. py:function:: paddle.fluid.layers.stack(x, axis=0)





该OP沿 ``axis`` 轴对输入 ``x`` 进行堆叠操作。

- 例1:

.. code-block:: text

    输入：
        x[0].shape = [1, 2]
        x[0].data = [ [1.0 , 2.0 ] ]
        x[1].shape = [1, 2]
        x[1].data = [ [3.0 , 4.0 ] ]
        x[2].shape = [1, 2]
        x[2].data = [ [5.0 , 6.0 ] ]

    参数：
        axis = 0 #沿着第0维对输入x进行堆叠操作。

    输出：
        Out.shape = [3, 1, 2]
        Out.data = [ [ [1.0, 2.0] ],
                    [ [3.0, 4.0] ],
                    [ [5.0, 6.0] ] ]


- 例2:

.. code-block:: text

    输入：
        x[0].shape = [1, 2]
        x[0].data = [ [1.0 , 2.0 ] ]
        x[1].shape = [1, 2]
        x[1].data = [ [3.0 , 4.0 ] ]
        x[2].shape = [1, 2]
        x[2].data = [ [5.0 , 6.0 ] ]

    参数：
        axis = 1 or axis = -2 #沿着第1维对输入进行堆叠操作。

    输出：
        Out.shape = [1, 3, 2]
        Out.data = [ [ [1.0, 2.0]
                      [3.0, 4.0]
                      [5.0, 6.0] ] ]

参数
::::::::::::

      - **x** (list(Variable)|tuple(Variable)) – 输入 x 可以是单个Tensor，或是多个Tensor组成的列表。如果 x 是一个列表，那么这些Tensor的维度必须相同。假设输入是N维Tensor :math:`[d_0,d_1,...,d_{n−1}]`，则输出变量的维度为N+1维 :math:`[d_0,d_1,...d_{axis-1},len(x),d_{axis}...,d_{n−1}]`。支持的数据类型：float32，float64，int32，int64。
      - **axis** (int，可选) – 指定对输入Tensor进行堆叠运算的轴，有效 ``axis`` 的范围是：:math:`[-(R+1), R+1)`，R是输入中第一个Tensor的rank。如果 ``axis`` < 0，则 :math:`axis=axis+rank(x[0])+1` 。axis默认值为0。

返回
::::::::::::
 堆叠运算后的Tensor，数据类型与输入Tensor相同。输出维度等于 :math:`rank(x[0])+1` 维。

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    # set batch size=None
    x1 = fluid.data(name='x1', shape=[None, 1, 2], dtype='int32')
    x2 = fluid.data(name='x2', shape=[None, 1, 2], dtype='int32')
    # stack Tensor list
    data = layers.stack([x1,x2]) # stack according to axis 0, data.shape=[2, None, 1, 2]

    data = layers.stack([x1,x2], axis=1) # stack according to axis 1, data.shape=[None, 2, 1, 2]
