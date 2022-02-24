.. _cn_api_vision_transforms_normalize:

normalize
-------------------------------

.. py:class:: paddle.vision.transforms.Normalize(mean=0.0, std=1.0, data_format='CHW', to_rgb=False, keys=None)

用均值和标准差归一化输入数据。

参数
:::::::::
    
    - mean (list|tuple) - 用于每个通道归一化的均值。
    - std (list|tuple) - 用于每个通道归一化的标准差值。
    - data_format (str, optional): 数据的格式，必须为 'HWC' 或 'CHW'。 默认值: 'CHW'。
    - to_rgb (bool, optional) - 是否转换为 ``rgb`` 的格式。默认值：False。

形状
:::::::::

    - img (PIL.Image|np.ndarray|paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回归一化后的图像数据。

返回
:::::::::

    计算 ``Normalize`` 的可调用对象。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import Normalize

    normalize = Normalize(mean=[127.5, 127.5, 127.5],
                          std=[127.5, 127.5, 127.5],
                          data_format='HWC')

    fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

    fake_img = normalize(fake_img)
    print(fake_img.shape)
    print(fake_img.max, fake_img.max)
    