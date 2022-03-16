.. _cn_api_vision_transforms_ToTensor:

ToTensor
-------------------------------

.. py:class:: paddle.vision.transforms.ToTensor(data_format='CHW', keys=None)

将 ``PIL.Image`` 或 ``numpy.ndarray`` 转换成 ``paddle.Tensor``。

将形状为 （H x W x C）的输入数据 ``PIL.Image`` 或 ``numpy.ndarray`` 转换为 (C x H x W)。
如果想保持形状不变，可以将参数 ``data_format`` 设置为 ``'HWC'``。

若输入数据形状为（H x W）， ``ToTensor`` 会将该数据的形状视为（H x W x 1）。

同时，如果输入的 ``PIL.Image`` 的 ``mode`` 是 ``(L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)`` 
其中一种，或者输入的 ``numpy.ndarray`` 数据类型是 'uint8'，那个会将输入数据从（0-255）的范围缩放到 
（0-1）的范围。其他的情况，则保持输入不变。


参数
:::::::::

    - data_format (str, optional): 返回张量的格式，必须为 'HWC' 或 'CHW'。 默认值: 'CHW'。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

形状
:::::::::

    - img (PIL.Image|numpy.ndarray) - 输入的图像数据，数据格式为'HWC'。
    - output (np.ndarray) - 返回的张量数据，根据参数 ``data_format``，张量的格式必须为 'HWC' 或 'CHW'。

返回
:::::::::

    计算 ``ToTensor`` 的可调用对象。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image

    import paddle.vision.transforms as T
    import paddle.vision.transforms.functional as F

    fake_img = Image.fromarray((np.random.rand(4, 5, 3) * 255.).astype(np.uint8))

    transform = T.ToTensor()

    tensor = transform(fake_img)
    
.. code-block:: python

Tensor(shape=[3, 4, 5], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [[[0.83137262, 0.01176471, 0.70980394, 0.67843139, 0.29411766],
         [0.72549021, 0.03137255, 0.01568628, 0.96470594, 0.87843144],
         [0.81960791, 0.05882353, 0.05098040, 0.28627452, 0.66666669],
         [0.28627452, 0.88235301, 0.21960786, 0.40392160, 0.79607850]],

        [[0.07058824, 0.85098046, 0.43921572, 0.92549026, 0.48235297],
         [0.41960788, 0.10588236, 0.41960788, 0.91764712, 0.55294120],
         [0.89019614, 0.56078434, 0.50196081, 0.40000004, 0.25882354],
         [0.43529415, 0.03137255, 0.71372551, 0.85882360, 0.86274517]],

        [[0.26274511, 0.98039222, 0.13725491, 0.48627454, 0.58039218],
         [0.37647063, 0.11372550, 0.93333340, 0.41960788, 0.69411767],
         [0.70196080, 0.78431380, 0.01960784, 0.44705886, 0.73333335],
         [0.73333335, 0.50588238, 0.32156864, 0.16862746, 0.34901962]]])
