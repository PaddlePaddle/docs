.. _cn_api_vision_transforms_flip:

flip
-------------------------------

.. py:function:: paddle.vision.transforms.flip(image, code)

根据翻转的类型对输入图像进行翻转。

参数
:::::::::

    - image (numpy.ndarray) - 输入的图像数据，形状为(H, W, C)。
    - code (int) - 支持的翻转类型，-1: 水平和垂直翻转，0: 垂直翻转，1: 水平翻转。

返回
:::::::::

    ``numpy.ndarray``，翻转后的图像数据。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import functional as F


    np.random.seed(5)
    fake_img = np.random.rand(224, 224, 3)

    # flip horizontally and vertically
    flip_img_1 = F.flip(fake_img, -1)
    
    # flip vertically
    flip_img_2 = F.flip(fake_img, 0)
    
    # flip horizontally
    flip_img_3 = F.flip(fake_img, 1)
    