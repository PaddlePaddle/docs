.. _cn_api_vision_transforms_BrightnessTransform:

BrightnessTransform
-------------------------------

.. py:class:: paddle.vision.transforms.BrightnessTransform(value)

调整图像的亮度。

参数
:::::::::

    - value (float) - 亮度调整范围大小，会从给定参数后的均匀分布[max(0，1 - brightness), 1 + brightness]中随机选择进行实际调整，可以是任何非负数。参数等于0时输出原始图像。

返回
:::::::::

    ``numpy ndarray``，调整亮度后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import BrightnessTransform


    transform = BrightnessTransform(0.4)
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)
        
    print(fake_img.shape)
    # (500, 500, 3)
    