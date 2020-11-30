.. _cn_api_vision_transforms_RandomVerticalFlip:

RandomVerticalFlip
-------------------------------

.. py:class:: paddle.vision.transforms.RandomVerticalFlip(prob=0.5, keys=None)

基于概率来执行图片的垂直翻转。

参数
:::::::::

    - prob (float) - 执行图片垂直翻转的概率，默认值为0.5。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

返回
:::::::::

    ``PIL.Image 或 numpy ndarray``，概率执行垂直翻转后的图像。

代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import RandomVerticalFlip

    transform = RandomVerticalFlip(224)

    fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.size)
