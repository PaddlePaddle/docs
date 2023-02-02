.. _cn_api_vision_transforms_ToTensor:

ToTensor
-------------------------------

.. py:class:: paddle.vision.transforms.ToTensor(data_format='CHW', keys=None)

将 ``PIL.Image`` 或 ``numpy.ndarray`` 转换成 ``paddle.Tensor``。

将形状为 （H x W x C）的输入数据 ``PIL.Image`` 或 ``numpy.ndarray`` 转换为 (C x H x W)。

若输入数据形状为（H x W）， ``ToTensor`` 会将该数据的形状视为（H x W x 1）。并且输出 Tensor 的形状是（1 x H x W）。

如果想保持形状不变，可以将参数 ``data_format`` 设置为 ``'HWC'``。

同时，如果输入的 ``PIL.Image`` 的 ``mode`` 是 ``(L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)``
其中一种，或者输入的 ``numpy.ndarray`` 数据类型是 'uint8'，那个会将输入数据从（0-255）的范围缩放到
（0-1）的范围。其他的情况，则保持输入不变。

在其他情况下， ``Tensor`` 将不缩放直接返回。

参数
:::::::::

    - **data_format** (str，可选) - 返回 Tensor 的格式，必须为 'HWC' 或 'CHW'。默认值：'CHW'。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|numpy.ndarray) - 输入的图像数据，数据格式为'HWC'。
    - output (np.ndarray) - 返回的 Tensor 数据，根据参数 ``data_format``，Tensor 的格式必须为 'HWC' 或 'CHW'。

返回
:::::::::

    计算 ``ToTensor`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.ToTensor
