.. _cn_api_fluid_layers_pad_constant_like:

pad_constant_like
-------------------------------

.. py:function:: paddle.fluid.layers.pad_constant_like(x, y, pad_value=0.0, name=None)




该OP使用 ``pad_value`` 填充 ``y``，填充到每个维度值的数量由x和y的形状而指定，((0，x.shape[0] - y.shape[0]), ..., (0, x.shape[i] - y.shape[i]), ..., (0, x.shape[n] - y.shape[n]))是每个维度填充的宽度，对于维度i，填充宽度 ``(0, x.shape[i] - y.shape[i])``，表示在y的第i维开头不填充，而在末尾填充 ``x.shape[i] - y.shape[i]`` 个位置。该OP要求y与x具有相同的秩，并且对每个维度i， ``y.shape[i] <= x.shape[i]`` 。

**示例**：

.. code-block:: text

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

    And
        pad_value = 0.

    Return:
        Out = [[[[35, 36, 37],
                 [ 0,  0,  0]],
                [[38, 39, 40],
                 [ 0,  0,  0]],
                [[41, 42, 43],
                 [ 0,  0,  0]]],
               [[[ 0,  0,  0], 
                 [ 0,  0,  0]],
                [[ 0,  0,  0], 
                 [ 0,  0,  0]],
                [[ 0,  0,  0], 
                 [ 0,  0,  0]]]]

        Out.shape = [2, 3, 2, 3]


参数
::::::::::::

          - **x** （Variable）- 多维Tensor
          - **y** （Variable）- 多维Tensor，与x具有相同的秩，而且对任意维度 ``i``，要求满足 ``y.shape[i] <= x.shape[i]``。数据类型为float32或float64
          - **pad_value** (float，可选) - 用于填充的常量值。默认值为0。
          - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
经过维度填充后的Tensor，与x具有相同的shape，与y具有相同的数据类型

返回类型
::::::::::::
  Variable

代码示例
::::::::::::

..  code-block:: python

    # x是秩为4的tensor, x.shape = (2, 3, 2, 3)
    # y是秩为4的tensor, y.shape = (1, 3, 1, 3)
    import paddle.fluid as fluid
    x = fluid.data(name='x', shape=[2,3,2,3], dtype='float32')
    y = fluid.data(name='y', shape=[1,3,1,3], dtype='float32')
    out = fluid.layers.pad_constant_like(x=x, y=y, pad_value=0.)
    # out是秩为4的tensor, out.shape = [2, 3 ,2 , 3]




