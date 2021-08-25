.. _cn_api_vision_transforms_CenterCrop:

CenterCrop
-------------------------------

.. py:class:: paddle.vision.transforms.CenterCrop(size, keys=None)

对输入图像进行裁剪，保持图片中心点不变。

参数
:::::::::

    - size (int|tuple) - 输出图像的形状大小。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

数据格式
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回裁剪后的图像数据。

返回
:::::::::

    ``CenterCrop`` 可调用对象。    

代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import CenterCrop

    transform = CenterCrop(224)

    fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.size)
    # out: (224, 224) width,height
