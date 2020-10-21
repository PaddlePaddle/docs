.. _cn_api_vision_transforms_HueTransform:

HueTransform
-------------------------------

.. py:class:: paddle.vision.transforms.HueTransform(value)

调整图像的色调。

参数
:::::::::

    - value (float) - 色调调整范围大小，，会从给定参数后的均匀分布[-hue, hue]中随机选择进行实际调整，参数值需要在0到0.5之间, 参数值为0时返回原始图像。

返回
:::::::::

    ``numpy ndarray``，调整色调后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import HueTransform


    transform = HueTransform(0.4)
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (500, 500, 3)
    