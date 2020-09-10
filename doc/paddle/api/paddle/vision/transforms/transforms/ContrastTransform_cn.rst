.. _cn_api_vision_transforms_ContrastTransform:

ContrastTransform
-------------------------------

.. py:class:: paddle.vision.transforms.ContrastTransform(value)

调整图像对比度。

参数
:::::::::

    - value (float) - 对比度调整范围大小，会从给定参数后的均匀分布[max(0，1 - contrast), 1 + contrast]中随机选择进行实际调整，不能是负数。参数值为0时返回原图像。

返回
:::::::::

    ``numpy ndarray``，调整对比度后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import ContrastTransform


    transform = ContrastTransform(0.4)
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)

    print(fake_img.shape)
    # (500, 500, 3)