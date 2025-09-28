.. _cn_api_paddle_vision_transforms_RandomCrop:

RandomCrop
-------------------------------

.. py:class:: paddle.vision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", keys=None)

在随机位置裁剪输入的图像。

参数
:::::::::

    - **size** (sequence|int) - 裁剪后的图片大小。如果 size 是一个 int 值，而不是(h, w)这样的序列，那么会做一个方形的裁剪(size, size)。
    - **padding** (int|sequence，可选) - 对图像四周外边进行填充，如果提供了长度为 4 的序列，则将其分别用于填充左边界，上边界，右边界和下边界。默认值：None，不填充。
    - **pad_if_needed** (boolean，可选) - 如果裁剪后的图像小于期望的大小时，是否对裁剪后的图像进行填充，以避免引发异常，默认值：False，保持初次裁剪后的大小，不填充。
    - **fill** (float|tuple，可选) - 用于填充的像素值。仅当 padding_mode 为 constant 时参数值有效。默认值：0。如果参数值是一个长度为 3 的元组，则会分别用于填充 R，G，B 通道。
    - **padding_mode** (string，可选) - 填充模式。支持：constant, edge, reflect 或 symmetric。默认值：constant。 ``constant`` 表示使用常量值进行填充，该值由 fill 参数指定。 ``edge`` 表示使用图像边缘像素值进行填充。 ``reflect`` 表示使用原图像的镜像值进行填充（不使用边缘上的值）；比如：使用该模式对 [1, 2, 3, 4] 的两端分别填充 2 个值，结果是 [3, 2, 1, 2, 3, 4, 3, 2]。 ``symmetric`` 表示使用原图像的镜像值进行填充（使用边缘上的值）；比如：使用该模式对 [1, 2, 3, 4] 的两端分别填充 2 个值，结果是 [2, 1, 1, 2, 3, 4, 4, 3]。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为 ``HWC`` 。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回随机裁剪后的图像数据。

返回
:::::::::

    计算 ``RandomCrop`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.RandomCrop
