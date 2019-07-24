.. _cn_api_fluid_layers_pad_constant_like:

pad_constant_like
-------------------------------

.. py:function:: paddle.fluid.layers.pad_constant_like(x, y, pad_value=0.0, name=None)

使用 ``pad_value`` 填充 ``Y`` ，填充到每个axis（轴）值的数量由X和Y的形不同而指定。（（0，shape_x_0 - shape_y_0），...（0，shape_x_n - shape_y_n ））是每个axis唯一pad宽度。输入应该是k维张量（k> 0且k <7）。

**实例如下**

::

    Given:
        X = [[[[ 0,  1,  2],
               [ 3,  4,  5]],
              [[ 6,  7,  8],
               [ 9, 10, 11]],
              [[12, 13, 14],
               [15, 16, 17]]],
             [[[18, 19, 20],
               [21, 22, 23]],
              [[24, 25, 26],
               [27, 28, 29]],
              [[30, 31, 32],
               [33, 34, 35]]]]
        X.shape = (2, 3, 2, 3)

        Y = [[[[35, 36, 37]],
              [[38, 39, 40]],
              [[41, 42, 43]]]]
        Y.shape = (1, 3, 1, 3)

参数：
          - **x** （Variable）- 输入Tensor变量。
          - **y** （Variable）- 输出Tensor变量。
          - **pad_value** (float) - 用于填充的常量值。
          - **name** （str | None） - 这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：填充张量（Tensor）变量

返回类型：  变量（Variable）

**示例代码**

..  code-block:: python

    # x是秩为4的tensor, x.shape = (2, 3, 2, 3)
    # y是秩为4的tensor, y.shape = (1, 3, 1, 3)
    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[2,3,2,3], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1,3,1,3], dtype='float32')
    out = fluid.layers.pad_constant_like(x=x, y=y, pad_value=0.)
    # out是秩为4的tensor, out.shape = [2, 3 ,2 , 3]




