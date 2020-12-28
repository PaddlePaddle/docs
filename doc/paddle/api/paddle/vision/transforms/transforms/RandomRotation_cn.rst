.. _cn_api_vision_transforms_RandomRotation:

RandomRotate
-------------------------------

.. py:class:: paddle.vision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None, fill=0, keys=None)

按角度旋转图像。

参数
:::::::::

    - degrees (sequence|float|int) - 旋转的角度度数范围。如果度数是数字而不是像（min，max）这样的序列，则会根据degrees参数值生成度数范围（-degrees，+degrees）。
    - interpolation (str, optional): 插值的方法. 如果这个参数没有设定或者输入图像为单通道，则该参数会根据使用的后端，被设置为 ``PIL.Image.NEAREST`` 或者 ``cv2.INTER_NEAREST`` 。 当使用 ``pil`` 作为后端时, 支持的插值方法如下: - "nearest": Image.NEAREST, - "bilinear": Image.BILINEAR, - "bicubic": Image.BICUBIC。当使用 ``cv2`` 作为后端时, 支持的插值方法如下: - "nearest": cv2.INTER_NEAREST, - "bilinear": cv2.INTER_LINEAR, - "bicubic": cv2.INTER_CUBIC
    - expand (bool，可选) - 是否要对旋转后的图片进行大小扩展，默认值: False，不进行扩展。当参数值为True时，会对图像大小进行扩展，让其能够足以容纳整个旋转后的图像。当参数值为False时，会按照原图像大小保留旋转后的图像。**这个扩展操作的前提是围绕中心旋转且没有平移。**
    - center (2-tuple，可选) - 旋转的中心点坐标，原点是图片左上角，默认值是图像的中心点。
    - fill (int，可选) - 对图像扩展时填充的值。默认值：0。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。
    
返回
:::::::::

    ``numpy ndarray``，随机旋转一定角度后的图像。

代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import RandomRotation

    transform = RandomRotation(90)

    fake_img = Image.fromarray((np.random.rand(200, 150, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.size)
    