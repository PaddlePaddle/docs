.. _cn_api_vision_transforms_RandomCrop:

RandomCrop
-------------------------------

.. py:class:: paddle.vision.transforms.RandomCrop(size, padding=0, pad_if_needed=False, keys=None)

在随机位置裁剪输入的图像。

参数
:::::::::

    - size (sequence|int) - 裁剪后的图片大小。如果size是一个int值，而不是(h, w)这样的序列，那么会做一个方形的裁剪(size, size)。
    - padding (int|sequence，可选) - 对图像四周外边进行填充，如果提供了长度为4的序列，则将其分别用于填充左边界，上边界，右边界和下边界。 默认值：0，不填充。
    - pad_if_needed (boolean，可选) - 如果裁剪后的图像小于期望的大小时，是否对裁剪后的图像进行填充，以避免引发异常，默认值：False，保持初次裁剪后的大小，不填充。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回随机裁剪后的图像数据。

返回
:::::::::

    计算 ``RandomCrop`` 的可调用对象。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import RandomCrop

    transform = RandomCrop(224)

    fake_img = Image.fromarray((np.random.rand(324, 300, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.size)
