.. _cn_api_vision_transforms_adjust_hue:

adjust_hue
-------------------------------

.. py:function:: paddle.vision.transforms.adjust_hue(img, hue_factor)

对输入图像进行色调的调整。

参数
:::::::::

    - img (PIL.Image|np.array) - 输入的图像。
    - hue_factor (float): 图像的色调通道的偏移量. 数值应在 ``[-0.5, 0.5]`` 。0.5和-0.5分别表示HSV空间中色相通道正向和负向完全反转，0表示没有调整色调。因此，-0.5和0.5都会给出一个带有互补色的图像，而0则会给出原始图像。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，调整后的图像。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

    fake_img = Image.fromarray(fake_img)

    converted_img = F.adjust_hue(fake_img, 0.4)
    print(converted_img.size)
        
