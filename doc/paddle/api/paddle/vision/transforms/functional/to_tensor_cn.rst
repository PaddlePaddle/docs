.. _cn_api_vision_transforms_to_tensor:

to_tensor
-------------------------------

.. py:function:: paddle.vision.transforms.to_tensor(pic, data_format='CHW')

将 ``PIL.Image`` 或 ``numpy.ndarray`` 转换成 ``paddle.Tensor``

参数
:::::::::

    - pic (PIL.Image|numpy.ndarray) - 输入的图像数据。
    - data_format (str, optional): 数据的格式，必须为 'HWC' 或 'CHW'。 默认值: 'CHW'。

返回
:::::::::

    ``paddle.Tensor``，转换后的数据。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import functional as F

    fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

    fake_img = Image.fromarray(fake_img)

    tensor = F.to_tensor(fake_img)
    print(tensor.shape)
    