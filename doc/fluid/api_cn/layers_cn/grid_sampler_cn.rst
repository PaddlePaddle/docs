.. _cn_api_fluid_layers_grid_sampler:

grid_sampler
-------------------------------

.. py:function::  paddle.fluid.layers.grid_sampler(x, grid, name=None)

该OP基于flow field网格的对输入X进行双线性插值采样。网格通常由affine_grid生成, shape为[N, H, W, 2]，是shape为[N, H, W]的采样点张量的(x, y)坐标。
其中，x坐标是对输入数据X的第四个维度(宽度维度)的索引，y坐标是第三维度(高维度)的索引，最终输出采样值为采样点的4个最接近的角点的双线性插值结果，输出张量的shape为[N, C, H, W]。

step 1：

  得到(x, y)网格坐标，缩放到[0,h -1/W-1]

.. code-block:: text

  grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1) grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

step 2：

  在每个[H, W]区域用网格(X, y)作为输入数据X的索引，并将双线性插值点值由4个最近的点表示。

.. code-block:: text

      wn ------- y_n ------- en
      |           |           |
      |          d_n          |
      |           |           |
     x_w --d_w-- grid--d_e-- x_e
      |           |           |
      |          d_s          |
      |           |           |
      ws ------- y_s ------- wn

    x_w = floor(x)              // west side x coord
    x_e = x_w + 1               // east side x coord
    y_n = floor(y)              // north side y coord
    y_s = y_s + 1               // south side y coord
    d_w = grid_x - x_w          // distance to west side
    d_e = x_e - grid_x          // distance to east side
    d_n = grid_y - y_n          // distance to north side
    d_s = y_s - grid_y          // distance to south side
    wn = X[:, :, y_n, x_w]      // north-west point value
    en = X[:, :, y_n, x_e]      // north-east point value
    ws = X[:, :, y_s, x_w]      // south-east point value
    es = X[:, :, y_s, x_w]      // north-east point value


    output = wn * d_e * d_s + en * d_w * d_s
           + ws * d_e * d_n + es * d_w * d_n

参数：
  - **x** (Variable): 输入张量，维度为 :math:`[N, C, H, W]` 的4-D Tensor，N为批尺寸，C是通道数，H是特征高度，W是特征宽度, 数据类型为float32或float64。
  - **grid** (Variable): 输入网格数据张量，维度为 :math:`[N, H, W, 2]` 的4-D Tensor，N为批尺寸，C是通道数，H是特征高度，W是特征宽度, 数据类型为float32或float64。
  - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回： Variable(Tensor): 输入X基于输入网格的双线性插值计算结果，维度为 :math:`[N, C, H, W]` 的4-D Tensor

返回类型：变量(Variable)，数据类型与 ``x`` 一致

**代码示例：**

.. code-block:: python

    # 一般与 affine_grid 组合使用
    x = fluid.layers.data(name='x', shape=[10, 32, 32], dtype='float32')
    theta = fluid.layers.data(name='theta', shape=[2, 3], dtype='float32')
    grid = fluid.layers.affine_grid(theta=theta, out_shape=[3, 10, 32, 32])
    out = fluid.layers.grid_sampler(x=x, grid=grid)










