.. _cn_api_vision_transforms_adjust_contrast:

adjust_contrast
-------------------------------

.. py:function:: paddle.vision.transforms.adjust_contrast(img, contrast_factor)

对输入图像进行对比度调整。

参数
:::::::::

    - **img** (PIL.Image|np.array|paddle.Tensor) - 输入的图像。
    - **contrast_factor** (float) - 调节图像对比度的多少，可以是任何非负数。参数等于 0 时输出纯灰色图像，参数等于 1 时输出原始图像，参数大于 1 时图像对比度增强，如参数等于 2 时图像对比度增强两倍。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray 或 paddle.Tensor``，调整后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.adjust_contrast
