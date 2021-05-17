.. _cn_api_vision_transforms_ToTensor:

ToTensor
-------------------------------

.. py:class:: paddle.vision.transforms.ToTensor(keys=None)

将 ``PIL.Image`` 或 ``numpy.ndarray`` 转换成 ``paddle.Tensor``。

将形状为 （H x W x C）的输入数据 ``PIL.Image`` 或 ``numpy.ndarray`` 转换为 (C x H x W)。
如果想保持形状不变，可以将参数 ``data_format`` 设置为 ``'HWC'``。

同时，如果输入的 ``PIL.Image`` 的 ``mode`` 是 ``(L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)`` 
其中一种，或者输入的 ``numpy.ndarray`` 数据类型是 'uint8'，那个会将输入数据从（0-255）的范围缩放到 
（0-1）的范围。其他的情况，则保持输入不变。


参数
:::::::::

    - data_format (str, optional): 返回张量的格式，必须为 'HWC' 或 'CHW'。 默认值: 'CHW'。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

返回
:::::::::

    ``paddle.Tensor``，变换后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image

    import paddle.vision.transforms as T
    import paddle.vision.transforms.functional as F

    fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

    transform = T.ToTensor()

    tensor = transform(fake_img)
    