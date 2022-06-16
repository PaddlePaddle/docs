.. _cn_api_fluid_layers_pad2d:

pad2d
-------------------------------

.. py:function::  paddle.fluid.layers.pad2d(input, paddings=[0, 0, 0, 0], mode='constant', pad_value=0.0, data_format='NCHW', name=None)




该OP依照 paddings 和 mode 属性对input进行2维 ``pad`` 。

参数
::::::::::::

  - **input** (Tensor) - 类型为float32的4-D Tensor，格式为 `[N, C, H, W]` 或 `[N, H, W, C]` 。
  - **paddings** (Tensor | List[int32]) - 填充大小。如果paddings是一个List，它必须包含四个整数 `[padding_top, padding_bottom, padding_left, padding_right]` 。
    如果paddings是Tensor，则是类型为int32 的1-D Tensor，形状是 `[4]`。默认值为 `[0,0,0,0]` 。
  - **mode** (str) - padding的三种模式，分别为 `'constant'` (默认)、 `'reflect'` 、 `'edge'` 。 `'constant'` 为填充常数 `pad_value` ， `'reflect'` 为填充以input边界值为轴的映射，`'edge'` 为填充input边界值。具体结果可见以下示例。默认值为 `'constant'` 。
  - **pad_value** (float32) - 以 `'constant'` 模式填充区域时填充的值。默认值为0.0。
  - **data_format** (str)  - 指定input的格式，可为 `'NCHW'` 和 `'NHWC'`，默认值为 `'NCHW'` 。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
返回
::::::::::::
Tensor，对input进行2维 pad 的结果，数据类型和input一样的4-D Tensor。

**示例**：

.. code-block:: text

      Input = [[[[1., 2., 3.],
                 [4., 5., 6.]]]]

      Case 0:
          paddings = [0, 1, 2, 3],
          mode = 'constant'
          pad_value = 0
          Out = [[[[0., 0., 1., 2., 3., 0., 0., 0.],
                   [0., 0., 4., 5., 6., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]]]]

      Case 1:
          paddings = [0, 1, 2, 1],
          mode = 'reflect'
          Out = [[[[3., 2., 1., 2., 3., 2.],
                   [6., 5., 4., 5., 6., 5.],
                   [3., 2., 1., 2., 3., 2.]]]]

      Case 2:
          paddings = [0, 1, 2, 1],
          mode = 'edge'
          Out = [[[[1., 1., 1., 2., 3., 3.],
                   [4., 4., 4., 5., 6., 6.],
                   [4., 4., 4., 5., 6., 6.]]]]



代码示例
::::::::::::

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn.functional as F

    # example 1
    x_shape = (1, 1, 3, 4)
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
    tensor_x = paddle.to_tensor(x)
    y = F.pad2d(tensor_x, paddings=[1, 2, 2, 1], pad_value=1, mode='constant')
    print(y.numpy())
    # [[[[ 1.  1.  1.  1.  1.  1.  1.]
    #    [ 1.  1.  1.  2.  3.  4.  1.]
    #    [ 1.  1.  5.  6.  7.  8.  1.]
    #    [ 1.  1.  9. 10. 11. 12.  1.]
    #    [ 1.  1.  1.  1.  1.  1.  1.]
    #    [ 1.  1.  1.  1.  1.  1.  1.]]]]

    # example 2
    x_shape = (1, 1, 2, 3)
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
    tensor_x = paddle.to_tensor(x)
    y = F.pad2d(tensor_x, paddings=[1, 1, 1, 1], mode='reflect')
    print(y.numpy())
    # [[[[5. 4. 5. 6. 5.]
    #    [2. 1. 2. 3. 2.]
    #    [5. 4. 5. 6. 5.]
    #    [2. 1. 2. 3. 2.]]]]
