.. _cn_api_fluid_layers_grid_sampler:

grid_sampler
-------------------------------

.. py:function::  paddle.fluid.layers.grid_sampler(x, grid, name=None)

该操作使用基于flow field网格的双线性插值对输入X进行采样，通常由affine_grid生成。

形状为(N、H、W、2)的网格是由两个形状均为(N、H、W)的坐标(grid_x grid_y)连接而成的。

其中，grid_x是输入数据x的第四个维度(宽度维度)的索引，grid_y是第三维度(高维)的索引，最终得到4个最接近的角点的双线性插值值。

step 1：

  得到(x, y)网格坐标，缩放到[0,h -1/W-1]

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
  - **x** (Variable): 输入数据，形状为[N, C, H, W]
  - **grid** (Variable): 输入网格张量，形状为[N, H, W, 2]
  - **name** (str, default None): 该层的名称

返回： **out** (Variable): 输入X基于输入网格的bilnear插值计算结果，形状为[N, C, H, W]

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name='x', shape=[3, 10, 32, 32], dtype='float32')
    theta = fluid.layers.data(name='theta', shape=[3, 2, 3], dtype='float32')
    grid = fluid.layers.affine_grid(theta=theta, out_shape=[3, 10, 32, 32]})
    out = fluid.layers.grid_sampler(x=x, grid=grid)










