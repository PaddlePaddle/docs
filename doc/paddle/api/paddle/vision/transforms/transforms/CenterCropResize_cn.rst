.. _cn_api_vision_transforms_CenterCropResize:

CenterCropResize
-------------------------------

.. py:class:: paddle.vision.transforms.CenterCropResize(size, crop_padding=32, interpolation=cv2.INTER_LINEAR)

通过填充将图像裁剪到图像中心，然后缩放尺寸。

参数
:::::::::

    - size (int|list|tuple) - 输出图像的形状大小。
    - crop_padding (int) - 中心裁剪时进行padding的大小。默认值: 32。
    - interpolation (int) - 调整图片大小时使用的插值模式。默认值: cv2.INTER_LINEAR。

返回
:::::::::

    ``numpy ndarray``，裁剪并调整尺寸后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import CenterCropResize


    transform = CenterCropResize(224)
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (224, 224, 3)
    