.. _cn_api_vision_transforms_adjust_brightness:

adjust_brightness
-------------------------------

.. py:function:: paddle.vision.transforms.adjust_brightness(img, brightness_factor)

对输入图像进行亮度值调整。

参数
:::::::::

    - img (PIL.Image|np.array) - 输入的图像。
    - brightness_factor (float): 调节图像亮度值的多少. 可以是任何非负数。参数等于1时输出原始图像。

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

    converted_img = F.adjust_brightness(fake_img, 0.4)
    print(converted_img.size)
        