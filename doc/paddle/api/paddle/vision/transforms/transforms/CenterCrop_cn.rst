.. _cn_api_vision_transforms_CenterCrop:

CenterCrop
-------------------------------

.. py:class:: paddle.vision.transforms.CenterCrop(output_size)

对输入图像进行裁剪，保持图片中心点不变。

参数
:::::::::

    - output_size (int|tuple) - 输出图像的形状大小。

返回
:::::::::

    ``numpy ndarray``，裁剪后的图像。    

代码示例
:::::::::
    
.. code-block:: python
    
    import numpy as np
    from paddle.vision.transforms import CenterCrop


    transform = CenterCrop(224)
    fake_img = np.random.rand(500, 500, 3).astype('float32')
    fake_img = transform(fake_img)
        
    print(fake_img.shape)
    # (224, 224, 3)