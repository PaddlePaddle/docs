.. _cn_api_vision_transforms_to_grayscale:

to_grayscale
-------------------------------

.. py:function:: paddle.vision.transforms.to_grayscale(img, num_output_channels=1)

将图像转换为灰度。

参数
:::::::::

    - **img** (PIL.Image|np.array) - 输入图像。
    - **num_output_channels** (int，可选) - 输出图像的通道数，默认值为 1，单通道。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，输入图像的灰度版本。

    - 如果 output_channels = 1：返回一个单通道图像。
    - 如果 output_channels = 3：返回一个 RBG 格式的 3 通道图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.to_grayscale
