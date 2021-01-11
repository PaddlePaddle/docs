.. _cn_api_vision_transforms_Resize:

Resize
-------------------------------

.. py:class:: paddle.vision.transforms.Resize(size, interpolation='bilinear', keys=None)

将输入数据调整为指定大小。

参数
:::::::::

    - size (int|list|tuple) - 输出图像大小。
            如果size是一个序列，例如（h，w），输出大小将与此匹配。
            如果size为int，图像的较小边缘将与此数字匹配，即如果 height > width，则图像将重新缩放为(size * height / width, size)。
    - interpolation (int|str, optional) - 插值的方法. 默认值: 'bilinear'. 
            当使用 ``pil`` 作为后端时, 支持的插值方法如下: 
            - "nearest": Image.NEAREST, 
            - "bilinear": Image.BILINEAR, 
            - "bicubic": Image.BICUBIC, 
            - "box": Image.BOX, 
            - "lanczos": Image.LANCZOS, 
            - "hamming": Image.HAMMING
            当使用 ``cv2`` 作为后端时, 支持的插值方法如下: : 
            - "nearest": cv2.INTER_NEAREST, 
            - "bilinear": cv2.INTER_LINEAR, 
            - "area": cv2.INTER_AREA, 
            - "bicubic": cv2.INTER_CUBIC, 
            - "lanczos": cv2.INTER_LANCZOS4
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。
返回
:::::::::

    ``numpy ndarray``，调整大小后的图像。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import Resize

    transform = Resize(size=224)

    fake_img = Image.fromarray((np.random.rand(100, 120, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.size)