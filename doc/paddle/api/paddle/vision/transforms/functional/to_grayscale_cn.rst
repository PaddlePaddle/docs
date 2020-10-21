.. _cn_api_vision_transforms_to_grayscale:

to_grayscale
-------------------------------

.. py:function:: paddle.vision.transforms.to_grayscale(img, num_output_channels=1)

将图像转换为灰度。

参数
:::::::::

    - img (numpy.ndarray) - 输入图像
    - num_output_channels (int，可选) - 输出图像的通道数，默认值为1，单通道。

返回
:::::::::

    ``numpy.ndarray``，输入图像的灰度版本。
        - 如果 output_channels == 1 : 返回一个单通道图像。
        - 如果 output_channels == 3 : 返回一个RBG格式的3通道图像。
    
代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from paddle.vision.transforms.functional import to_grayscale

    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = to_grayscale(fake_img)
    
    print(fake_img.shape)
    # (500, 500, 1)
    