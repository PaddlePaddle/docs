.. _cn_api_paddle_vision_ops_image_decode_random_crop:

image_decode_random_crop
-------------------------------

.. py:function:: paddle.vision.ops.image_decode_random_crop(x, num_threads=2, host_memory_padding=0, device_memory_padding=0, data_format='NCHW', aspect_ratio_min=3./4., aspect_ratio_max=4./3., area_min=0.08, area_max=1.0, num_attempts=10, name=None)

将一个批次的JPEG图像通过Nvjpeg多线程解码为3维的Tensor并做随机裁剪，默认解码格式为RGBI，更多信息请见https://docs.nvidia.com/cuda/nvjpeg/index.html

输出Tensor数据类型为uint8，值在0到255之间。

此API仅能在PaddlePaddle GPU版本中使用

参数
:::::::::
    - x (Tensor) - 包含JPEG图像位数据的1维uint8 Tensor列表。
    - num_threads (int) - 解码子线程数，默认为2.
    - host_memory_padding (int) - Nvjpeg解码分配CUDA Pinned内存的填充值，默认为0.
    - device_memory_padding (int) - Nvjpeg解码分配CUDA内存的填充值，默认为0.
    - data_format (str) - 输出图像的格式，如果为NCHW，则输出图像形状为(channel, height, width)，如果为NHWC，则输出图像形状为(height, width, channel)，默认为NCHW
    - aspect_ratio_min (float) - 随机图像裁剪框的最小纵横比，默认为3/4。
    - aspect_ratio_max (float) - 随机图像裁剪框的最大纵横比，默认为4/3。
    - area_min (float) - 随机图像裁剪框的最小面积比率，默认为0.08。
    - area_max (float) - 随机图像裁剪框的最大面积比率，默认为1.0。
    - num_attempts (int) - 随机图像裁剪的最大尝试次数，默认为10。
    - name (str，可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

返回
:::::::::
    形状为(channel, width, height)解码图像数组

代码示例
:::::::::

..  code-block:: python

    import cv2
    import paddle
    import numpy as np

    fake_img = (np.random.random(
            (400, 300, 3)) * 255).astype('uint8')

    cv2.imwrite('fake.jpg', fake_img)

    # only support GPU version
    if not paddle.get_device() == 'cpu':
        img_bytes = paddle.vision.ops.read_file('fake.jpg')
        imgs = paddle.vision.ops.image_decode_random_crop([img_bytes])

        print(imgs[0].shape)
