.. _cn_api_vision_transforms_adjust_brightness:

adjust_brightness
-------------------------------

.. py:function:: paddle.vision.transforms.adjust_brightness(img, brightness_factor)

对输入图像进行亮度值调整。

参数
:::::::::

    - img (PIL.Image|np.array|paddle.Tensor) - 输入的图像。
    - brightness_factor (float) - 调节图像亮度值的多少，可以是任何非负数。参数等于0时输出黑色图像，参数等于1时输出原始图像，参数大于1时输出图像亮度增强，如参数等于2时图像亮度增强两倍。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray 或 paddle.Tensor``，调整后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.adjust_brightness


