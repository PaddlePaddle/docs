.. _cn_api_vision_transforms_resize:

resize
-------------------------------

.. py:function:: paddle.vision.transforms.resize(img, size, interpolation=cv2.INTER_LINEAR)

将输入数据调整为指定大小。

参数
:::::::::

    - img (numpy.ndarray) - 输入数据，可以是(H, W, C)形状的图像或遮罩。
    - size (int|tuple) - 输出图像大小。
            如果size是一个序列，例如（h，w），输出大小将与此匹配。
            如果size为int，图像的较小边缘将与此数字匹配，即如果 height > width，则图像将重新缩放为(size * height / width, size)。
    - interpolation (int，可选) - 调整图片大小时使用的插值模式。默认值: cv2.INTER_LINEAR。

返回
:::::::::

    ``numpy.ndarray``，调整大小后的图像数据。

代码示例
:::::::::

.. code-block:: python
    
    import numpy as np
    from paddle.vision.transforms import functional as F


    fake_img = np.random.rand(256, 256, 3)
    fake_img_1 = F.resize(fake_img, 224)

    print(fake_img_1.shape)
    # (224, 224, 3)

    fake_img_2 = F.resize(fake_img, (200, 150))

    print(fake_img_2.shape)
    # (200, 150, 3)
        