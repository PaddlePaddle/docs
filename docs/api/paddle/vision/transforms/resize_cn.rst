.. _cn_api_vision_transforms_resize:

resize
-------------------------------

.. py:function:: paddle.vision.transforms.resize(img, size, interpolation='bilinear')

将输入数据调整为指定大小。

参数
:::::::::

    - **img** (numpy.ndarray|PIL.Image) - 输入数据，可以是(H, W, C)形状的图像或遮罩。
    - **size** (int|tuple) - 输出图像大小。如果 size 是一个序列，例如（h，w），输出大小将与此匹配。如果 size 为 int，图像的较小边缘将与此数字匹配，即如果 height > width，则图像将重新缩放为(size * height / width, size)。
    - **interpolation** (int|str，可选) - 插值的方法，默认值：'bilinear'。

        - 当使用 ``pil`` 作为后端时，支持的插值方法如下

            + "nearest": Image.NEAREST,
            + "bilinear": Image.BILINEAR,
            + "bicubic": Image.BICUBIC,
            + "box": Image.BOX,
            + "lanczos": Image.LANCZOS,
            + "hamming": Image.HAMMING。

        - 当使用 ``cv2`` 作为后端时，支持的插值方法如下

            + "nearest": cv2.INTER_NEAREST,
            + "bilinear": cv2.INTER_LINEAR,
            + "area": cv2.INTER_AREA,
            + "bicubic": cv2.INTER_CUBIC,
            + "lanczos": cv2.INTER_LANCZOS4。

返回
:::::::::

``PIL.Image`` 或 ``numpy.ndarray``，调整大小后的图像数据。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.resize
