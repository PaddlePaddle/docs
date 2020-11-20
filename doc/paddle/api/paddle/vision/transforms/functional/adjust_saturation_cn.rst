.. _cn_api_vision_transforms_adjust_saturation:

adjust_saturation
-------------------------------

.. py:function:: paddle.vision.transforms.adjust_saturation(img, saturation_factor)

对输入图像进行饱和度调整。

参数
:::::::::

    - img (PIL.Image|np.array) - 输入的图像。
    - saturation_factor (float): 调节图像饱和度的多少. 可以是任何非负数。参数等于1时输出原始图像。

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

    converted_img = F.adjust_saturation(fake_img, 0.4)
    print(converted_img.size)
        