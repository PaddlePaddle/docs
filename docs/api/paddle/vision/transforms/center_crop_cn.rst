.. _cn_api_vision_transforms_center_crop:

center_crop
-------------------------------

.. py:function:: paddle.vision.transforms.center_crop(img, output_size)

对输入图像进行中心裁剪。

参数
:::::::::

    - **img** (PIL.Image|np.array) - 用于裁剪的图像，（0,0）表示图像的左上角。
    - **output_size** (sequence|list) - 要裁剪的矩形框的大小：(height, width)。如果是 ``int`` 值，则所有方向按照这个值裁剪。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，裁剪后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.center_crop
