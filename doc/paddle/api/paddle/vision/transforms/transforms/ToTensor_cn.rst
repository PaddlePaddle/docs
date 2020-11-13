.. _cn_api_vision_transforms_ToTensor:

ToTensor
-------------------------------

.. py:class:: paddle.vision.transforms.ToTensor(keys=None)

将 ``PIL.Image`` 或 ``numpy.ndarray`` 转换成 ``paddle.Tensor``

参数
:::::::::

    - data_format (str, optional): 数据的格式，必须为 'HWC' 或 'CHW'。 默认值: 'CHW'。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

返回
:::::::::

    ``paddle.Tensor``，变换后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image

    import paddle.vision.transforms as T
    import paddle.vision.transforms.functional as F

    fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

    transform = T.ToTensor()

    tensor = transform(fake_img)
    