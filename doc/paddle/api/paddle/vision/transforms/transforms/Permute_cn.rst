.. _cn_api_vision_transforms_Permute:

Permute
-------------------------------

.. py:class:: paddle.vision.transforms.Permute(mode="CHW", to_rgb=True)

将输入的图像数据更改为目标格式。例如，大多数数据预处理是使用HWC格式的图片，而神经网络可能使用CHW模式输入张量。

.. note::
    输入图像应为HWC格式的numpy.ndarray。 

参数
:::::::::

    - mode (str) - 输出图像的格式，默认值为CHW（图像通道-图像高度-图像宽度）。
    - to_rgb (bool) - 将BGR格式图像转换为RGB格式，默认值为True，启用此项转换。

返回
:::::::::

    ``numpy ndarray``，更改格式后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import Permute


    transform = Permute()
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (3, 500, 500)