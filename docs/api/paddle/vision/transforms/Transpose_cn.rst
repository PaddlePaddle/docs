.. _cn_api_vision_transforms_Transpose:

Transpose
-------------------------------

.. py:class:: paddle.vision.transforms.Transpose(order=(2, 0, 1), keys=None)

将输入的图像数据更改为目标格式。例如，大多数数据预处理是使用HWC格式的图片，而神经网络可能使用CHW模式输入张量。
输出的图片是numpy.ndarray的实例。

参数
:::::::::

    - order (list|tuple, optional) - 目标的维度顺序. Default: (2, 0, 1)。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform``. 默认值: None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (np.ndarray|Paddle.Tensor) - 返回更改格式后的数组或张量。如果输入是``PIL.Image``，输出将会自动转换为``np.ndarray``。

返回
:::::::::

    计算 ``Transpose`` 的可调用对象。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import Transpose

    transform = Transpose()

    fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.shape)
