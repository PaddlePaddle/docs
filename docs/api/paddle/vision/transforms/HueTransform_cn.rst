.. _cn_api_vision_transforms_HueTransform:

HueTransform
-------------------------------

.. py:class:: paddle.vision.transforms.HueTransform(value)

调整图像的色调。

参数
:::::::::

    - value (float) - 色调调整范围大小，，会从给定参数后的均匀分布[-hue, hue]中随机选择进行实际调整，参数值需要在0到0.5之间, 参数值为0时返回原始图像。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

数据格式
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回调整色调后的图像数据。

返回
:::::::::

    ``HueTransform`` 可调用对象。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import HueTransform

    transform = HueTransform(0.4)

    fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    