.. _cn_api_nn_functional_grid_sample:

grid_sample
-------------------------------

.. py:function::  paddle.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True, name=None):




该 OP 基于 flow field 网格的对输入 X 进行双线性插值采样。网格通常由 affine_grid 生成，当输入X为4维时，网格shape 为[N, H, W, 2]，是 shape 为[N, H, W]的采样点张量的(x, y)坐标。
其中，x 坐标是对输入数据 X 的第四个维度(宽度维度)的索引，y 坐标是第三维度(高维度)的索引，最终输出采样值为采样点的 4 个最接近的角点的双线性插值结果，输出张量的 shape 为[N, C, H, W]。
当输入X为5维时，网格shape为[N, D, H, W, 3]，是shape为[N, D, H, W]的采样点张量的(x, y, z)坐标。其中，x坐标是对输入数据X的第五个维度(宽度维度)的索引，y坐标是第四维度
(高度维度)的索引，z坐标是对输入数据X的第三个维度(深度维度)的索引，最终输出采样值为采样点的8个最接近的角点的双线性插值结果，输出张量的shape为[N, C, D, H, W]。

step 1：

  得到(x, y)网格坐标，缩放到[0,h -1/W-1]

.. code-block:: text

  grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1) grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

step 2：

  在每个[H, W]区域用网格(X, y)作为输入数据 X 的索引，并将双线性插值点值由 4 个最近的点表示。

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

参数
::::::::::::

  - **x** (Tensor)：输入张量，维度为 :math:`[N, C, H, W]` 的 4-D Tensor或维度为 :math:`[N, C, D, H, W]` 的5-D Tensor，N 为批尺寸，C 是通道数，
                    D是特征深度，H 是特征高度，W 是特征宽度，数据类型为 float32 或 float64。
  - **grid** (Tensor)：输入网格数据张量，维度为 :math:`[N, H, W, 2]` 的 4-D Tensor或维度为 :math:`[N, D, H, W, 3]` 的5-D Tensor，N 为批尺寸，
                       H 是特征高度，D是特征深度，W 是特征宽度，数据类型为 float32 或 float64。
  - **mode** (str, optional)：插值方式，可以为 'bilinear' 或者 'nearest'。默认值：'bilinear'。
  - **padding_mode** (str, optional) 当原来的索引超过输入的图像大小时的填充方式。可以为 'zeros', 'reflection' 和 'border'。默认值：'zeros'。
  - **align_corners** (bool, optional)：一个可选的 bool 型参数，如果为 True，则将输入和输出张量的 4 个或 8 个角落像素的中心对齐，并保留角点像素的值。默认值：True。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，输入 X 基于输入网格的双线性插值计算结果，维度为 :math:`[N, C, H, W]` 的 4-D Tensor或维度为 :math:`[N, C, D, H, W]` 的5-D Tensor，
数据类型与 ``x`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.grid_sample
