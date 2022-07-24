.. _cn_api_fluid_layers_im2sequence:

im2sequence
-------------------------------


.. py:function:: paddle.fluid.layers.im2sequence(input, filter_size=1, stride=1, padding=0, input_image_size=None, out_stride=1, name=None)




该OP使用 `filter` 扫描输入的Tensor并将输入Tensor转换成序列，返回值的 `shape={input.batch_size * output_height * output_width, filter_size_height* filter_size_width * input.channels}`。返回值的timestep的个数为 `output_height * output_width`，每个timestep的维度是 `filter_size_height* filter_size_width * input.channels`。其中 `output_height` 和 `output_width` 由以下式计算：


.. math::
    output\_height = 1 + \frac{padding\_up + padding\_down + input\_height - filter\_size\_height + stride\_height-1}{stride\_height} \\
    output\_width = 1 + \frac{padding\_left + padding\_right + input\_width - filter\_size\_width + stride\_width-1}{stride\_width}

其中符号的意义如下所示。

参数
::::::::::::

  - **input** （Variable）- 类型为float32的4-D Tensor，格式为 `[N, C, H, W]`。公式中 `input_height` 和 `input_width` 分别代表输入的高和宽。
  - **filter_size** (int32 | List[int32]) - 滤波器大小。如果 `filter_size` 是一个List，它必须包含两个整数 `[filter_size_height, filter_size_width]`。如果 `filter_size` 是一个int32，则滤波器大小是 `[filter_size, filter_size]`，默认值为1。
  - **stride** (int32 | List[int32]) - 步长大小。如果stride是一个List，它必须包含两个整数 `[stride_height,stride_width]`。如果stride是一个int32，则步长大小是 `[stride, stride]`，默认值为1。
  - **padding** (int32 | List[int32]) - 填充大小。如果padding是一个List，它可以包含四个整数 `[padding_up, padding_left, padding_down, padding_right]`，当包含两个整数 `[padding_height, padding_width]` 时，可展开为 `[padding_height, padding_width, padding_height, padding_width]`。如果padding是一个int，可展开为 `[padding, padding, padding, padding]`。默认值为0。
  - **input_image_size** (Variable，可选) - 2-D Tensor，输入图像的实际大小，它的维度为 `[batchsize，2]`。当该参数不为None时，可用于batch inference。默认值为None。
  - **out_stride** (int32 | List[int32]) - 输出步长。只有当input_image_size不为None时才有效。如果out_stride是List，它必须包含 `[out_stride_height, out_stride_width]`，如果out_stride是int32，则可展开为 `[out_stride, out_stride]`，默认值为1。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 数据类型为float32, `shape` 为 `{batch_size * output_height * output_width, filter_size_height * filter_size_width * input.channels}` 的 2-D LodTensor。

返回类型
::::::::::::
 Variable

::

  Given:

    x = [[[[ 6.  2.  1.]
      [ 8.  3.  5.]
      [ 0.  2.  6.]]

        [[ 2.  4.  4.]
         [ 6.  3.  0.]
         [ 6.  4.  7.]]]

       [[[ 6.  7.  1.]
         [ 5.  7.  9.]
         [ 2.  4.  8.]]

        [[ 1.  2.  1.]
         [ 1.  3.  5.]
         [ 9.  0.  8.]]]]

    x.dims = {2, 2, 3, 3}

    And:

    filter = [2, 2]
    stride = [1, 1]
    padding = [0, 0]

    Then:

    output.data = [[ 6.  2.  8.  3.  2.  4.  6.  3.]
                   [ 2.  1.  3.  5.  4.  4.  3.  0.]
                   [ 8.  3.  0.  2.  6.  3.  6.  4.]
                   [ 3.  5.  2.  6.  3.  0.  4.  7.]
                   [ 6.  7.  5.  7.  1.  2.  1.  3.]
                   [ 7.  1.  7.  9.  2.  1.  3.  5.]
                   [ 5.  7.  2.  4.  1.  3.  9.  0.]
                   [ 7.  9.  4.  8.  3.  5.  0.  8.]]

    output.dims = {8, 8}

    output.lod = [[4, 4]]


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.im2sequence