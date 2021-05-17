.. _cn_api_vision_transforms_ColorJitter:

ColorJitter
-------------------------------

.. py:class:: paddle.vision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0, keys=None)

随机调整图像的亮度，对比度，饱和度和色调。

参数
:::::::::

    - brightness(float) - 亮度调整范围大小，会从给定参数后的均匀分布[max(0，1 - brightness), 1 + brightness]中随机选择进行实际调整，不能是负数。
    - contrast(float) - 对比度调整范围大小，，会从给定参数后的均匀分布[max(0，1 - contrast), 1 + contrast]中随机选择进行实际调整，不能是负数。
    - saturation(float) - 饱和度调整范围大小，，会从给定参数后的均匀分布[max(0，1 - saturation), 1 + saturation]中随机选择进行实际调整，不能是负数。
    - hue(float) - 色调调整范围大小，，会从给定参数后的均匀分布[-hue, hue]中随机选择进行实际调整，参数值需要在0到0.5之间。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

返回
:::::::::

    ``numpy ndarray``，调整亮度、对比度、饱和度和色调后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import ColorJitter

    transform = ColorJitter(0.4, 0.4, 0.4, 0.4)

    fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    