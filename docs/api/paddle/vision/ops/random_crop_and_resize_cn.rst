.. _cn_api_paddle_vision_ops_random_crop_and_resize:

random_crop_and_resize
-------------------------------

.. py:function:: paddle.vision.ops.random_crop_and_resize(x, size, aspect_ratio_min=3./4., aspect_ratio_max=4./3., area_min=0.08, area_max=1.0, num_attempts=10, interp_method='bilinear', align_corners=True, align_mode=1, data_format='NCHW', name=None)

此OP实现GPU版本的 paddle.vision.transforms.RandomResizedCrop ，详细信息请见 https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/RandomResizedCrop_cn.html#randomresizedcrop

此API仅能在PaddlePaddle GPU版本中使用

参数
:::::::::
    - x (List[Tensor]) - 包含JPEG图像位数据的1维uint8 Tensor列表。
    - size (int | List(int)) - 输出图像的大小，格式为(height, width)
    - aspect_ratio_min (float) - 随机图像裁剪框的最小纵横比，默认为3/4。
    - aspect_ratio_max (float) - 随机图像裁剪框的最大纵横比，默认为4/3。
    - area_min (float) - 随机图像裁剪框的最小面积比率，默认为0.08。
    - area_max (float) - 随机图像裁剪框的最大面积比率，默认为1.0。
    - num_attempts (int) - 随机图像裁剪的最大尝试次数，默认为10。
    - interp_method (str) - 缩放图像时的插值方式，支持双线性插值'bilinear'和'nearest'最近领插值，默认为'bilinear'。
    - align_corners (bool, 可选) - 一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。 默认值为False。
    - align_mode (int, 可选) - 双线性插值的可选项。 可以是 '0' 代表src_idx = scale *（dst_indx + 0.5）-0.5；如果为'1' ，代表src_idx = scale * dst_index。默认值：0。
    - data_format (str) - 输出图像的格式，如果为NCHW，则输出图像形状为(channel, height, width)，如果为NHWC，则输出图像形状为(height, width, channel)，默认为NCHW
    - name (str，可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

返回
:::::::::
    输出Tensor为4维，形状为(batch_size, channels, size_h, size_w)或(batch_size, size_h, size_w, channels)，数据类型为uint8或者float32

代码示例
:::::::::

..  code-block:: python

    import paddle

    # only support GPU version
    if not paddle.get_device() == 'cpu':
        data = paddle.randn(shape=[3, 256, 256])
        data = paddle.cast(data, dtype='uint8')
        out = paddle.vision.ops.random_crop_and_resize([data], size=224)

        print(out.shape)
