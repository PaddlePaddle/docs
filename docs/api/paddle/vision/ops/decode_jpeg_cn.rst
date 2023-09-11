.. _cn_api_paddle_vision_ops_decode_jpeg:

decode_jpeg
-------------------------------

.. py:function:: paddle.vision.ops.decode_jpeg(x, mode='unchanged', name=None)

将 JPEG 图像解码为三维 RGB Tensor 或者 一维灰度 Tensor，解码格式可选，输出 Tensor 的值为 uint8 类型，介于 0 到 255 之间。


参数
:::::::::

    - **x** (Tensor) - 包含 JPEG 图像原始字节的一维 uint8 Tensor。
    - **mode** (str，可选) - 转换图像模式选择，默认值为 'unchanged'。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::

    具有形状（imge_channels、image_height、image_width）的解码图像 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.vision.ops.decode_jpeg
