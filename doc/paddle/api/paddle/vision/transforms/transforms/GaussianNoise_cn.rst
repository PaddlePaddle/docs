.. _cn_api_vision_transforms_GaussianNoise:

GaussianNoise
-------------------------------

.. py:class:: paddle.vision.transforms.GaussianNoise(mean=0.0, std=1.0)

基于给定的均值和标准差来产生高斯噪声，并将随机高斯噪声添加到输入数据。

参数
:::::::::

    - mean (float) - 用于产生噪声的高斯平均值。
    - std (float) - 用于产生噪声的高斯标准偏差。

返回
:::::::::

    ``numpy ndarray``，增加高斯噪声后的图像。    

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import GaussianNoise


    transform = GaussianNoise()
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (500, 500, 3)
    