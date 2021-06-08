.. _cn_api_vision_transforms_hflip:

hflip
-------------------------------

.. py:function:: paddle.vision.transforms.hflip(img)

对输入图像进行水平翻转。

参数
:::::::::

    - img (PIL.Image|numpy.ndarray) - 输入的图像。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，翻转后的图像数据。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

    fake_img = Image.fromarray(fake_img)

    flpped_img = F.hflip(fake_img)
    print(flpped_img.size)
    