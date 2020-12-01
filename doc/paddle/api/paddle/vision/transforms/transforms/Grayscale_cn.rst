.. _cn_api_vision_transforms_Grayscale:

Grayscale
-------------------------------

.. py:class:: paddle.vision.transforms.Grayscale(num_output_channels=1, keys=None)

将图像转换为灰度。

参数
:::::::::

    - num_output_channels (int，可选) - 输出图像的通道数，参数值为1或3。默认值：1。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

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
    from paddle.vision.transforms import Grayscale

    transform = Grayscale()

    fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(np.array(fake_img).shape)
    