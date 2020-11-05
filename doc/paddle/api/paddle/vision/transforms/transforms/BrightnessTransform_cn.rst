.. _cn_api_vision_transforms_BrightnessTransform:

BrightnessTransform
-------------------------------

.. py:class:: paddle.vision.transforms.BrightnessTransform(value)

调整图像的亮度。

参数
:::::::::

    - value (float) - 亮度调整范围大小，会从给定参数后的均匀分布[max(0，1 - brightness), 1 + brightness]中随机选择进行实际调整，可以是任何非负数。参数等于0时输出原始图像。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。
返回
:::::::::

    ``PIL.Image 或 numpy ndarray``，调整亮度后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import BrightnessTransform

    transform = BrightnessTransform(0.4)

    fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    