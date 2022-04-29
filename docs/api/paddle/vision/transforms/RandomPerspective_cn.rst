.. _cn_api_vision_transforms_Perspective:

Perspective
-------------------------------

.. py:class:: paddle.vision.transforms.Perspective(prob=0.5, distortion_scale=0.5, interpolation='nearest', fill=0, keys=None)

依据distortion_scale失真程度参数的范围，按照一定概率对图像进行透视变换。

参数
:::::::::

    - prob (float) - 进行透视变换的概率，默认是0.5。
    - distortion_scale (float) - 控制失真程度参数，范围为0到1。默认值为0.5。
    - interpolation (str, optional): 插值的方法。
        如果这个参数没有设定或者输入图像为单通道，则该参数会根据使用的后端，被设置为 ``PIL.Image.NEAREST`` 或者 ``cv2.INTER_NEAREST``。
        当使用 ``pil`` 作为后端时, 支持的插值方法如下:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC
        当使用 ``cv2`` 作为后端时, 支持的插值方法如下:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "bicubic": cv2.INTER_CUBIC
    - fill (int，可选) - 对图像扩展时填充的值。默认值：0。
    - keys (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值: None。
    
形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回随机透视变换后的图像数据。

返回
:::::::::

    计算 ``Perspective`` 的可调用对象。

代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import Perspective

    transform = Perspective(prob=1.0, distortion_scale=0.9)

    fake_img = Image.fromarray((np.random.rand(200, 150, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.size)
    
