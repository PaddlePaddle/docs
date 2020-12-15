.. _cn_api_vision_transforms_Pad:

Pad
-------------------------------

.. py:class:: paddle.vision.transforms.Pad(padding, fill=0, padding_mode='constant', keys=None)

使用特定的填充模式和填充值来对输入图像进行填充。

参数
:::::::::

    - padding (int|tuple) - 在图像每个边框上填充。
            如果提供单个int值，则用于填充图像所有边。
            如果提供长度为2的元组，则分别为图像左/右和顶部/底部进行填充。
            如果提供了长度为4的元组，则分别按照左，上，右和下的顺序为图像填充。
    - fill (int|tuple) - 用于填充的像素值，仅当padding_mode为constant时传递此参数，默认使用0来进行每个像素的填充。
            如果参数值是一个长度为3的元组，则会分别用于填充R，G，B通道。
    - padding_mode (string) - 填充模式，支持：constant, edge, reflect 或 symmetric，默认值：constant，使用fill参数值填充。
            ``constant`` 表示使用fill参数来指定一个值进行填充。
            ``edge`` 表示在图像边缘填充最后一个值。
            ``reflect`` 表示用原图像的反向图片填充（不重复使用边缘上的值）。比如使用这个模式对 ``[1, 2, 3, 4]``的两端分别填充2个值，最后结果是 ``[3, 2, 1, 2, 3, 4, 3, 2]``。
            ``symmetric`` 表示用原图像的反向图片填充（重复使用边缘上的值）。比如使用这个模式对 ``[1, 2, 3, 4]``的两端分别填充2个值，最后结果是 ``[2, 1, 1, 2, 3, 4, 4, 3]``。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

返回
:::::::::

    ``PIL.Image 或 numpy ndarray``，填充后的图像。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import Pad
    
    transform = Pad(2)
    
    fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))
    
    fake_img = transform(fake_img)
    print(fake_img.size)
    