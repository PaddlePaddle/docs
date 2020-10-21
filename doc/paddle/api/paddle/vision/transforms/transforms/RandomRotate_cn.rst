.. _cn_api_vision_transforms_RandomRotate:

RandomRotate
-------------------------------

.. py:class:: paddle.vision.transforms.RandomRotate(degrees, interpolation=cv2.INTER_LINEAR, expand=False, center=None)

按角度旋转图像。

参数
:::::::::

    - degrees (sequence|float|int) - 旋转的角度度数范围。如果度数是数字而不是像（min，max）这样的序列，则会根据degrees参数值生成度数范围（-degrees，+degrees）。
    - interpolation (int，可选) - 调整图片大小时使用的插值模式。默认值: cv2.INTER_LINEAR。
    - expand (bool，可选): 是否要对旋转后的图片进行大小扩展，默认值: False，不进行扩展。
            当参数值为True时，会对图像大小进行扩展，让其能够足以容纳整个旋转后的图像。
            当参数值为False时，会按照原图像大小保留旋转后的图像。
            **这个扩展操作的前提是围绕中心旋转且没有平移。**
    - center (2-tuple，可选) - 旋转的中心点坐标，原点是图片左上角，默认值是图像的中心点。
    
返回
:::::::::

    ``numpy ndarray``，随机旋转一定角度后的图像。

代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from paddle.vision.transforms import RandomRotate


    transform = RandomRotate(90)
    np.random.seed(5)
    fake_img = np.random.rand(500, 400, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (500, 400, 3)
    