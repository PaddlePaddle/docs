.. _cn_api_vision_transforms_normalize:

normalize
-------------------------------

.. py:function:: paddle.vision.transforms.normalize(pic, data_format='CHW')

将 ``PIL.Image`` 或 ``numpy.ndarray`` 转换成 ``paddle.Tensor``

参数
:::::::::
    
    - img (PIL.Image|np.array|paddle.Tensor) - 用于归一化的数据。
    - mean (list|tuple) - 用于每个通道归一化的均值。
    - std (list|tuple) - 用于每个通道归一化的标准差值。
    - data_format (str, optional): 数据的格式，必须为 'HWC' 或 'CHW'。 默认值: 'CHW'。
    - to_rgb (bool, optional) - 是否转换为 ``rgb`` 的格式。默认值：False。

返回
:::::::::

    ``numpy array 或 paddle.Tensor``，归一化后的图像。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

    fake_img = Image.fromarray(fake_img)

    mean = [127.5, 127.5, 127.5]
    std = [127.5, 127.5, 127.5]

    normalized_img = F.normalize(fake_img, mean, std, data_format='HWC')
    print(normalized_img.max(), normalized_img.min())
    