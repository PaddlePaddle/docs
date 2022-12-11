.. _cn_api_vision_transforms_adjust_hue:

adjust_hue
-------------------------------

.. py:function:: paddle.vision.transforms.adjust_hue(img, hue_factor)

对输入图像进行色调的调整。

通过将图像转换为 HSV 和周期性调整色调通道（H）的强度。然后将图像转换为原始图像模式。

``hue_factor`` 是 H 通道的位移量，必须在 ``[-0.5, 0.5]`` 之间。

参数
:::::::::

    - **img** (PIL.Image|np.array|paddle.Tensor) - 输入的图像。
    - **hue_factor** (float) - 图像的色调通道的偏移量。数值应在 ``[-0.5, 0.5]`` 。0.5 和-0.5 分别表示 HSV 空间中色相通道正向和负向完全反转，0 表示没有调整色调。因此，-0.5 和 0.5 都会给出一个带有互补色的图像，而 0 则会给出原始图像。

返回
:::::::::

    ``PIL.Image`` 或 ``numpy.ndarray`` 或 ``paddle.Tensor``，调整后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.adjust_hue
