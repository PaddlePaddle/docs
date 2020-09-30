.. _cn_api_vision_transforms_Normalize:

Normalize
-------------------------------

.. py:class:: paddle.vision.transforms.Normalize(mean=0.0, std=1.0)

图像归一化处理，支持两种方式：
1. 用统一的均值和标准差值对图像的每个通道进行归一化处理；
2. 对每个通道指定不同的均值和标准差值进行归一化处理。

计算过程：

``output[channel] = (input[channel] - mean[channel]) / std[channel]``

参数
:::::::::

    - mean (int|float|list) - 用于每个通道归一化的均值。
    - std (int|float|list) - 用于每个通道归一化的标准差值。

返回
:::::::::

    ``numpy ndarray``，归一化后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import Normalize


    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    fake_img = np.random.rand(3, 500, 500).astype('float32')
    fake_img = normalize(fake_img)

    print(fake_img.shape)
    # (3, 500, 500)