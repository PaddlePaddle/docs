.. _cn_api_vision_transforms_SaturationTransform:

SaturationTransform
-------------------------------

.. py:class:: paddle.vision.transforms.SaturationTransform(value)

调整图像的饱和度

参数
:::::::::

    - value (float) - 饱和度的调整数值，非负数，当参数值为0时返回原始图像。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回调整饱和度后的图像数据。

返回
:::::::::

    计算 ``SaturationTransform`` 的可调用对象。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import SaturationTransform

    transform = SaturationTransform(0.4)

    fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    