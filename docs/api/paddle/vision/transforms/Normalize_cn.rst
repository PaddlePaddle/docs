.. _cn_api_vision_transforms_Normalize:

Normalize
-------------------------------

.. py:class:: paddle.vision.transforms.Normalize(mean=0.0, std=1.0, data_format='CHW', to_rgb=False, keys=None)

图像归一化处理，支持两种方式：
1. 用统一的均值和标准差值对图像的每个通道进行归一化处理；
2. 对每个通道指定不同的均值和标准差值进行归一化处理。

计算过程：

``output[channel] = (input[channel] - mean[channel]) / std[channel]``

参数
:::::::::

    - mean (int|float|list) - 用于每个通道归一化的均值。
    - std (int|float|list) - 用于每个通道归一化的标准差值。
    - data_format (str, optional): 数据的格式，必须为 'HWC' 或 'CHW'。 默认值: 'CHW'。
    - to_rgb (bool, optional) - 是否转换为 ``rgb`` 的格式。默认值：False。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform``. 默认值: None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
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
    