.. _cn_api_vision_transforms_perspective:

perspective
-------------------------------

.. py:function:: paddle.vision.transforms.perspective(img, startpoints, endpoints, interpolation='nearest', fill=0)

对图像进行透视变换。

参数
:::::::::

    - img (PIL.Image|numpy.ndarray|paddle.Tensor) - 输入图像。
    - startpoints (list(list(float))) - 在原图上的四个角（左上、右上、右下、左下）的坐标。
    - endpoints (list(list(float))) - 在变换后的图上的四个角（左上、右上、右下、左下）的坐标。
    - interpolation (str, 可选): 插值的方法。
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

返回
:::::::::

    ``PIL.Image 或 numpy ndarray 或 paddle.Tensor``，透视变换后的图像。

代码示例
:::::::::
    
.. code-block:: python
        
    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')
    fake_img = Image.fromarray(fake_img)

    startpoints = [[0, 0], [33, 0], [33, 25], [0, 25]]
    endpoints = [[3, 2], [32, 3], [30, 24], [2, 25]]

    perspectived_img = F.perspective(fake_img, startpoints, endpoints)
    print(perspectived_img.size)
    
