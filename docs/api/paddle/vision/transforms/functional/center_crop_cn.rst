.. _cn_api_vision_transforms_center_crop:

center_crop
-------------------------------

.. py:function:: paddle.vision.transforms.center_crop(img, output_size)

对输入图像进行中心裁剪。

参数
:::::::::

    - img (PIL.Image|np.array) - 用于裁剪的图像。
    - output_size (int|list|tuple): 要裁剪的矩形框的大小：(height, width)。如果是 ``int`` 值，则所有方向按照这个值裁剪。

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

    cropped_img = F.center_crop(fake_img, (150, 100))
    print(cropped_img.size)
    # out: (100, 150) width,height