.. _cn_api_paddle_vision_ops_image_decode_random_crop:

image_decode_random_crop
-------------------------------

.. py:function:: paddle.vision.ops.image_decode_random_crop(x, num_threads=2, host_memory_padding=0, device_memory_padding=0, aspect_ratio_min=3./4., aspect_ratio_max=4./3., area_min=0.08, area_max=1.0, num_attempts=10, name=None)

将一个批次的JPEG图像通过Nvjpeg多线程解码为3维的Tensor并做随机裁剪，默认解码格式为RGBI，更多信息请见https://docs.nvidia.com/cuda/nvjpeg/index.html

输出Tensor数据类型为uint8，值在0到255之间。

.. note::
  此API仅能在PaddlePaddle GPU版本中使用

参数
:::::::::
    - **x** (List[Tensor]) - 包含JPEG图像位数据的1维uint8 Tensor列表。
    - **num_threads** (int, 可选) - 解码子线程数，默认为2.
    - **host_memory_padding** (int, 可选) - Nvjpeg解码分配CUDA Pinned内存的填充值，如果大于0，会预分配对应大小的CUDA Pinned内存作为缓存，设置合理时能避免在执行过程中重复分配内存。默认为0.
    - **device_memory_padding** (int, 可选) - Nvjpeg解码分配CUDA内存的填充值，如果大于0，会预分配对应大小的CUDA Pinned内存作为缓存，设置合理时能避免在执行过程中重复分配内存。默认为0.
    - **aspect_ratio_min** (float, 可选) - 随机图像裁剪框的最小纵横比，默认为3/4。
    - **aspect_ratio_max** (float, 可选) - 随机图像裁剪框的最大纵横比，默认为4/3。
    - **area_min** (float, 可选) - 随机图像裁剪框的最小面积比率，默认为0.08。
    - **area_max** (float, 可选) - 随机图像裁剪框的最大面积比率，默认为1.0。
    - **num_attempts** (int, 可选) - 随机图像裁剪的最大尝试次数，若超出尝试次数不满足纵横比、面积限制，则不做裁剪。须为正整数，默认为10。
    - **name** (str，可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

返回
:::::::::
    形状为(width, height, channels)解码图像数组

代码示例
:::::::::

..  code-block:: python

COPY-FROM: paddle.vision.ops.image_decode_random_crop:code-example
