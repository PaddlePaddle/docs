.. _cn_api_vision_transforms_RandomVerticalFlip:

RandomVerticalFlip
-------------------------------

.. py:class:: paddle.vision.transforms.RandomVerticalFlip(prob=0.5)

基于概率来执行图片的垂直翻转。

参数
:::::::::

    - prob (float) - 执行图片垂直翻转的概率，默认值为0.5。

返回
:::::::::

    ``numpy ndarray``，概率执行垂直翻转后的图像。

代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from paddle.vision.transforms import RandomVerticalFlip
    
    
    transform = RandomVerticalFlip()
    np.random.seed(5)
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (500, 500, 3)