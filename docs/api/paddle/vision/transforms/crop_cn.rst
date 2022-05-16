.. _cn_api_vision_transforms_crop:

crop
-------------------------------

.. py:function:: paddle.vision.transforms.crop(img, top, left, height, width)

对输入图像进行裁剪。

参数
:::::::::

    - img (PIL.Image|np.array) - 用于裁剪的图像.
    - top (int) - 要裁剪的矩形框左上方的坐标点的垂直方向的值.
    - left (int) - 要裁剪的矩形框左上方的坐标点的水平方向的值.
    - height (int) - 要裁剪的矩形框的高度值.
    - width (int) - 要裁剪的矩形框的宽度值.

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，裁剪后的图像。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

    fake_img = Image.fromarray(fake_img)

    cropped_img = F.crop(fake_img, 56, 150, 200, 100)
    print(cropped_img.size)
    