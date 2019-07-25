.. _cn_api_fluid_layers_im2sequence:

im2sequence
-------------------------------

.. py:function:: paddle.fluid.layers.im2sequence(input, filter_size=1, stride=1, padding=0, input_image_size=None, out_stride=1, name=None)

从输入张量中提取图像张量，与im2col相似，shape={input.batch_size * output_height * output_width, filter_size_H * filter_size_W * input.通道}。这个op使用filter / kernel扫描图像并将这些图像转换成序列。一个图片展开后的timestep的个数为output_height * output_width，其中output_height和output_width由下式计算:


.. math::
                        output\_size=1+\frac{(2∗padding+img\_size−block\_size+stride-1)}{stride}

每个timestep的维度为 :math:`block\_y * block\_x * input.channels` 。

参数:
  - **input** （Variable）- 输入张量，格式为[N, C, H, W]
  - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个tuple，它必须包含两个整数(filter_size_H, filter_size_W)。否则，过滤器将是一个方阵。
  - **stride** (int|tuple) - 步长大小。如果stride是一个元组，它必须包含两个整数(stride_H、stride_W)。否则，stride_H = stride_W = stride。默认:stride = 1。
  - **padding** (int|tuple) - 填充大小。如果padding是一个元组，它可以包含两个整数(padding_H, padding_W)，这意味着padding_up = padding_down = padding_H和padding_left = padding_right = padding_W。或者它可以使用(padding_up, padding_left, padding_down, padding_right)来指示四个方向的填充。否则，标量填充意味着padding_up = padding_down = padding_left = padding_right = padding Default: padding = 0。
  - **input_image_size** (Variable) - 输入包含图像的实际大小。它的维度为[batchsize，2]。该参数可有可无，是用于batch上的预测。
  - **out_stride** (int|tuple) - 通过CNN缩放图像。它可有可无，只有当input_image_size不为空时才有效。如果out_stride是tuple，它必须包含(out_stride_H, out_stride_W)，否则，out_stride_H = out_stride_W = out_stride。
  - **name** (int) - 该layer的名称，可以忽略。

返回： LoDTensor shape为{batch_size * output_height * output_width, filter_size_H * filter_size_W * input.channels}。如果将输出看作一个矩阵，这个矩阵的每一行都是一个序列的step。

返回类型: output

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


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[3, 32, 32],
                             dtype='float32')
    output = fluid.layers.im2sequence(
        input=data, stride=[1, 1], filter_size=[2, 2])










