.. _cn_api_vision_transforms_SaturationTransform:

SaturationTransform
-------------------------------

.. py:class:: paddle.vision.transforms.SaturationTransform(value)

调整图像的饱和度

参数
:::::::::

    - value (float) - 饱和度的调整数值，非负数，当参数值为0时返回原始图像。

返回
:::::::::

    ``numpy ndarray``，调整饱和度后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import SaturationTransform


    transform = SaturationTransform(0.4)
    np.random.seed(5)
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (500, 500, 3)