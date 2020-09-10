.. _cn_api_vision_transforms_Resize:

Resize
-------------------------------

.. py:class:: paddle.vision.transforms.Resize(size, interpolation=cv2.INTER_LINEAR)

将输入数据调整为指定大小。

参数
:::::::::

    - size (int|list|tuple) - 输出图像大小。
            如果size是一个序列，例如（h，w），输出大小将与此匹配。
            如果size为int，图像的较小边缘将与此数字匹配，即如果 height > width，则图像将重新缩放为(size * height / width, size)。
    - interpolation (int，可选) - 调整图片大小时使用的插值模式。默认值: cv2.INTER_LINEAR。

返回
:::::::::

    ``numpy ndarray``，调整大小后的图像。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import Resize
    
    
    transform = Resize(size=224)
    np.random.seed(5)
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)
    
    print(fake_img.shape)
    # (224, 224, 3)