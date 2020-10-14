.. _cn_api_vision_transforms_Grayscale:

Grayscale
-------------------------------

.. py:class:: paddle.vision.transforms.Grayscale(output_channels=1)

将图像转换为灰度。

参数
:::::::::

    - output_channels (int) - 输出图像的通道数，参数值为1或3。

返回
:::::::::

    ``numpy.ndarray``，输入图像的灰度版本。
        - 如果 output_channels == 1 : 返回一个单通道图像。
        - 如果 output_channels == 3 : 返回一个RBG格式的3通道图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import Grayscale


    transform = Grayscale()
    fake_img = np.random.rand(500, 400, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (500, 400, 1)
    