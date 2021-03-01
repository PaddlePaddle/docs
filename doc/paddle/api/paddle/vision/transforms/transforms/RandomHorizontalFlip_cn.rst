.. _cn_api_vision_transforms_RandomHorizontalFlip:

RandomHorizontalFlip
-------------------------------

.. py:class:: paddle.vision.transforms.RandomHorizontalFlip(prob=0.5, keys=None)

基于概率来执行图片的水平翻转。

参数
:::::::::

    - prob (float) - 图片执行水平翻转的概率，取值范围为[0, 1], 默认值为0.5。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

返回
:::::::::

    ``PIL.Image 或 numpy ndarray``，概率执行水平翻转后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import RandomHorizontalFlip

    transform = RandomHorizontalFlip(0.5)

    fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.size)
