.. _cn_api_vision_transforms_crop:

crop
-------------------------------

.. py:function:: paddle.vision.transforms.crop(img, top, left, height, width)

对输入图像进行裁剪。

参数
:::::::::

    - **img** (PIL.Image|np.array) - 用于裁剪的图像。
    - **top** (int) - 要裁剪的矩形框左上方的坐标点的垂直方向的值。
    - **left** (int) - 要裁剪的矩形框左上方的坐标点的水平方向的值。
    - **height** (int) - 要裁剪的矩形框的高度值。
    - **width** (int) - 要裁剪的矩形框的宽度值。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，裁剪后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.crop
