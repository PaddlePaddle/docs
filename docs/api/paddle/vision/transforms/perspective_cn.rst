.. _cn_api_paddle_vision_transforms_perspective:

perspective
-------------------------------

.. py:function:: paddle.vision.transforms.perspective(img, startpoints, endpoints, interpolation='nearest', fill=0)

对图像进行透视变换。

参数
::::::::::::
    - **img** (PIL.Image|numpy.ndarray|paddle.Tensor) - 输入图像。
    - **startpoints** (list(list(float))) - 在原图上的四个角（左上、右上、右下、左下）的坐标。
    - **endpoints** (list(list(float))) - 在变换后的图上的四个角（左上、右上、右下、左下）的坐标。
    - **interpolation** (str，可选) - 插值的方法。
        如果这个参数没有设定或者输入图像为单通道，则该参数会根据使用的后端，被设置为 ``PIL.Image.NEAREST`` 或者 ``cv2.INTER_NEAREST`` 。
        当使用 ``pil`` 作为后端时, 支持的插值方法如下:
            - "nearest": Image.NEAREST
            - "bilinear": Image.BILINEAR
            - "bicubic": Image.BICUBIC
        当使用 ``cv2`` 作为后端时, 支持的插值方法如下:
            - "nearest": cv2.INTER_NEAREST
            - "bilinear": cv2.INTER_LINEAR
            - "bicubic": cv2.INTER_CUBIC
    - **fill** (int|list|tuple，可选) - 对图像扩展时填充的像素值，默认值： 0 ，如果只设定一个数字则所有通道上像素值均为该值。

返回
::::::::::::

    ``PIL.Image 或 numpy ndarray 或 paddle.Tensor`` ，透视变换后的图像。

代码示例
::::::::::::

COPY-FROM: paddle.vision.transforms.perspective
