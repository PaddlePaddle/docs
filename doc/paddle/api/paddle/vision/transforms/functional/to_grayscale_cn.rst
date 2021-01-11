.. _cn_api_vision_transforms_to_grayscale:

to_grayscale
-------------------------------

.. py:function:: paddle.vision.transforms.to_grayscale(img, num_output_channels=1)

将图像转换为灰度。

参数
:::::::::

    - img (PIL.Image|np.array) - 输入图像。
    - num_output_channels (int，可选) - 输出图像的通道数，默认值为1，单通道。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，输入图像的灰度版本。
    - 如果 output_channels == 1 : 返回一个单通道图像。
    - 如果 output_channels == 3 : 返回一个RBG格式的3通道图像。
    
代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

    fake_img = Image.fromarray(fake_img)

    gray_img = F.to_grayscale(fake_img)
    print(gray_img.size)
    