.. _cn_api_vision_transforms_to_tensor:

to_tensor
-------------------------------

.. py:function:: paddle.vision.transforms.to_tensor(pic, data_format='CHW')

将 ``PIL.Image`` 或 ``numpy.ndarray`` 转换成 ``paddle.Tensor``。

将形状为 （H x W x C）的输入数据 ``PIL.Image`` 或 ``numpy.ndarray`` 转换为 (C x H x W)。
如果想保持形状不变，可以将参数 ``data_format`` 设置为 ``'HWC'``。

同时，如果输入的 ``PIL.Image`` 的 ``mode`` 是 ``(L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)`` 
其中一种，或者输入的 ``numpy.ndarray`` 数据类型是 'uint8'，那个会将输入数据从（0-255）的范围缩放到 
（0-1）的范围。其他的情况，则保持输入不变。

参数
:::::::::

    - pic (PIL.Image|numpy.ndarray) - 输入的图像数据。
    - data_format (str, optional): 返回的张量的格式，必须为 'HWC' 或 'CHW'。 默认值: 'CHW'。

返回
:::::::::

    ``paddle.Tensor``，转换后的数据。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

    fake_img = Image.fromarray(fake_img)

    tensor = F.to_tensor(fake_img)
    print(tensor.shape)
    